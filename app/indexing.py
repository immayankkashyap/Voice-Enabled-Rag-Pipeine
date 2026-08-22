"""Paired FAISS storage for native-MRL and full Jina embeddings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import Chunk

_MANIFEST_VERSION = 1
_MRL_INDEX_FILE = "mrl.faiss"
_FULL_INDEX_FILE = "full.faiss"
_FULL_VECTORS_FILE = "full_embeddings.npy"
_CHUNKS_FILE = "chunks.json"
_MANIFEST_FILE = "manifest.json"


def _faiss_module() -> Any:
    # Delay the native import so callers can initialize Torch/Jina first. On
    # macOS, importing FAISS before Torch can crash the process at model load.
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - dependency installation path
        raise RuntimeError("FaissVectorStore requires faiss-cpu") from exc
    return faiss


@dataclass(frozen=True, slots=True)
class IndexDimensions:
    """The native Jina v3 dimensions used by the two retrieval stages."""

    full: int = 1_024
    mrl: int = 128

    def __post_init__(self) -> None:
        if self.full <= 0 or self.mrl <= 0:
            raise ValueError("Index dimensions must be positive")
        if self.mrl >= self.full:
            raise ValueError("MRL dimension must be smaller than full dimension")


def _matrix(value: np.ndarray, *, dimension: int, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError(f"{name} must have shape (n, {dimension})")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    return np.ascontiguousarray(matrix)


def _normalized_rows(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.ascontiguousarray(value.copy(), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} contains a zero vector")
    matrix /= norms
    return matrix


def _normalized_query(value: np.ndarray, *, dimension: int, name: str) -> np.ndarray:
    query = np.asarray(value, dtype=np.float32).reshape(-1)
    if query.shape != (dimension,) or not np.all(np.isfinite(query)):
        raise ValueError(f"{name} must be one finite {dimension}-dimensional vector")
    norm = float(np.linalg.norm(query))
    if norm == 0:
        raise ValueError(f"{name} cannot be a zero vector")
    return np.ascontiguousarray(query / norm, dtype=np.float32)


class FaissVectorStore:
    """Own two aligned FAISS indices and exact full-vector rerank storage.

    ``jinaai/jina-embeddings-v3-hf`` is natively Matryoshka-trained at 128 and
    1,024 dimensions. Consequently, taking the first 128 dimensions and
    renormalizing is native MRL inference here, not an untrained truncation
    approximation. A different model must set ``native_mrl=False`` so that
    limitation is explicit in the saved manifest.
    """

    def __init__(
        self,
        dimensions: IndexDimensions | None = None,
        *,
        hnsw_m: int = 32,
        ef_construction: int = 80,
        ef_search: int = 64,
        native_mrl: bool = True,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        if hnsw_m <= 0 or ef_construction <= 0 or ef_search <= 0:
            raise ValueError("HNSW settings must be positive")
        self.dimensions = dimensions or IndexDimensions()
        self.hnsw_m = hnsw_m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.native_mrl = native_mrl
        self.provenance = dict(provenance or {})

        faiss = _faiss_module()
        self.mrl_index = faiss.IndexHNSWFlat(
            self.dimensions.mrl, self.hnsw_m, faiss.METRIC_INNER_PRODUCT
        )
        self.mrl_index.hnsw.efConstruction = self.ef_construction
        self.mrl_index.hnsw.efSearch = self.ef_search
        self.full_index = faiss.IndexFlatIP(self.dimensions.full)
        self.full_embeddings = np.empty((0, self.dimensions.full), dtype=np.float32)
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], full_embeddings: np.ndarray) -> None:
        """Normalize and add aligned full/MRL vectors to both indices."""

        matrix = _matrix(
            full_embeddings,
            dimension=self.dimensions.full,
            name="full_embeddings",
        )
        if not chunks or len(chunks) != matrix.shape[0]:
            raise ValueError("chunks and full_embeddings must be non-empty and aligned")
        existing_ids = {chunk.id for chunk in self.chunks}
        incoming_ids = [chunk.id for chunk in chunks]
        if len(set(incoming_ids)) != len(incoming_ids) or existing_ids.intersection(
            incoming_ids
        ):
            raise ValueError("Chunk IDs must be unique across the index")

        full = _normalized_rows(matrix, name="full_embeddings")
        # Matryoshka inference truncates the leading dimensions, followed by
        # normalization for cosine/IP search at the chosen nested dimension.
        mrl = _normalized_rows(
            full[:, : self.dimensions.mrl], name="MRL-truncated embeddings"
        )
        self.mrl_index.add(mrl)
        self.full_index.add(full)
        self.full_embeddings = np.ascontiguousarray(
            np.concatenate((self.full_embeddings, full), axis=0), dtype=np.float32
        )
        self.chunks.extend(chunks)
        self._validate_alignment()

    def search_mrl(
        self, query_embedding: np.ndarray, candidate_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return approximate candidate row IDs and MRL inner-product scores."""

        if candidate_k <= 0:
            raise ValueError("candidate_k must be positive")
        if not self.chunks:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape == (self.dimensions.full,):
            query = query[: self.dimensions.mrl]
        query = _normalized_query(
            query, dimension=self.dimensions.mrl, name="MRL query embedding"
        )
        count = min(candidate_k, len(self.chunks))
        scores, indices = self.mrl_index.search(query[None, :], count)
        valid = indices[0] >= 0
        return indices[0][valid].astype(np.int64), scores[0][valid].astype(np.float32)

    def rerank_full(
        self,
        query_embedding: np.ndarray,
        candidate_ids: np.ndarray,
        final_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exactly rerank candidate rows with aligned 1,024-d vectors."""

        if final_k <= 0:
            raise ValueError("final_k must be positive")
        ids = np.asarray(candidate_ids, dtype=np.int64).reshape(-1)
        if not len(ids):
            return ids, np.empty(0, dtype=np.float32)
        if np.any(ids < 0) or np.any(ids >= len(self.chunks)):
            raise ValueError("candidate_ids contain an out-of-range row")
        query = _normalized_query(
            query_embedding,
            dimension=self.dimensions.full,
            name="full query embedding",
        )
        scores = self.full_embeddings[ids] @ query
        order = np.argsort(-scores, kind="stable")[: min(final_k, len(ids))]
        return ids[order], np.asarray(scores[order], dtype=np.float32)

    def save(self, output_dir: Path) -> None:
        """Persist both indices, exact vectors, chunks, and build provenance."""

        if not self.chunks:
            raise ValueError("Cannot save an empty vector store")
        self._validate_alignment()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        faiss = _faiss_module()
        faiss.write_index(self.mrl_index, str(output_dir / _MRL_INDEX_FILE))
        faiss.write_index(self.full_index, str(output_dir / _FULL_INDEX_FILE))
        np.save(
            output_dir / _FULL_VECTORS_FILE, self.full_embeddings, allow_pickle=False
        )
        (output_dir / _CHUNKS_FILE).write_text(
            json.dumps(
                [chunk.model_dump(mode="json") for chunk in self.chunks],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": _MANIFEST_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dimensions": asdict(self.dimensions),
            "vector_count": len(self.chunks),
            "native_mrl": self.native_mrl,
            "mrl_index_type": "IndexHNSWFlat",
            "full_index_type": "IndexFlatIP",
            "hnsw": {
                "m": self.hnsw_m,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
            },
            "files": {
                "mrl_index": _MRL_INDEX_FILE,
                "full_index": _FULL_INDEX_FILE,
                "full_vectors": _FULL_VECTORS_FILE,
                "chunks": _CHUNKS_FILE,
            },
            "provenance": self.provenance,
        }
        (output_dir / _MANIFEST_FILE).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, input_dir: Path) -> FaissVectorStore:
        """Load and validate an offline-built index bundle."""

        input_dir = Path(input_dir)
        manifest = json.loads((input_dir / _MANIFEST_FILE).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != _MANIFEST_VERSION:
            raise ValueError("Unsupported FAISS bundle manifest version")
        dimensions = IndexDimensions(**manifest["dimensions"])
        hnsw = manifest["hnsw"]
        store = cls(
            dimensions,
            hnsw_m=int(hnsw["m"]),
            ef_construction=int(hnsw["ef_construction"]),
            ef_search=int(hnsw["ef_search"]),
            native_mrl=bool(manifest["native_mrl"]),
            provenance=manifest.get("provenance", {}),
        )
        faiss = _faiss_module()
        store.mrl_index = faiss.read_index(str(input_dir / _MRL_INDEX_FILE))
        store.full_index = faiss.read_index(str(input_dir / _FULL_INDEX_FILE))
        store.mrl_index.hnsw.efSearch = store.ef_search
        store.full_embeddings = _matrix(
            np.load(input_dir / _FULL_VECTORS_FILE, allow_pickle=False),
            dimension=dimensions.full,
            name="persisted full_embeddings",
        )
        raw_chunks = json.loads((input_dir / _CHUNKS_FILE).read_text(encoding="utf-8"))
        store.chunks = [Chunk.model_validate(value) for value in raw_chunks]
        if int(manifest["vector_count"]) != len(store.chunks):
            raise ValueError("Manifest vector count does not match chunk metadata")
        store._validate_alignment()
        return store

    def _validate_alignment(self) -> None:
        count = len(self.chunks)
        if (
            self.mrl_index.d != self.dimensions.mrl
            or self.full_index.d != self.dimensions.full
            or self.mrl_index.ntotal != count
            or self.full_index.ntotal != count
            or self.full_embeddings.shape != (count, self.dimensions.full)
        ):
            raise ValueError("FAISS indices, exact vectors, and chunks are misaligned")
