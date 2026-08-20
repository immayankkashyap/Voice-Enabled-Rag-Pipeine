import asyncio
import time
import sys
import os
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.schemas import RAGRequest
from app.main import process_rag_query
from app.indexing import vector_store

async def run_benchmark():
    # Setup dummy index
    print("Setting up dummy index for benchmark...")
    from scripts.build_index import build_index
    build_index()
    
    queries = [
        "What is the capital of India?",
        "How does FAISS work?",
        "Explain retrieval augmented generation.",
        "Tell me about late chunking.",
        "What is MRL?"
    ]
    
    print(f"Running benchmark with {len(queries)} queries...")
    latencies = {
        'total': [],
        'retrieval': [],
        'relevance_check': [],
        'generation': [],
        'groundedness_check': []
    }
    
    for i, query in enumerate(queries):
        req = RAGRequest(query=query)
        
        start_time = time.time()
        res = await process_rag_query(req)
        total_latency = (time.time() - start_time) * 1000
        
        latencies['total'].append(total_latency)
        latencies['retrieval'].append(res.latencies.get('retrieval', 0))
        latencies['relevance_check'].append(res.latencies.get('relevance_check', 0))
        latencies['generation'].append(res.latencies.get('generation', 0))
        latencies['groundedness_check'].append(res.latencies.get('groundedness_check', 0))
        
        print(f"Query {i+1} completed in {total_latency:.2f}ms")

    print("\n--- Benchmark Results (Latency in ms) ---")
    metrics = {}
    for stage, lats in latencies.items():
        if not lats: continue
        metrics[stage] = {
            'p50': np.percentile(lats, 50),
            'p70': np.percentile(lats, 70),
            'p100': np.percentile(lats, 100)
        }
        print(f"{stage.upper()}:")
        print(f"  P50:  {metrics[stage]['p50']:.2f} ms")
        print(f"  P70:  {metrics[stage]['p70']:.2f} ms")
        print(f"  P100: {metrics[stage]['p100']:.2f} ms")
        
    os.makedirs('data', exist_ok=True)
    with open('data/benchmark_report.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved benchmark report to data/benchmark_report.json")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
