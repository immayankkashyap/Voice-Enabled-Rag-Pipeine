"use strict";

const TARGET_SAMPLE_RATE = 16_000;
const BATCH_SAMPLES = 320;
const MAX_FRAME_BYTES = 3_200;
const READY_TIMEOUT_MS = 8_000;
const FLUSH_TIMEOUT_MS = 750;

const startButton = document.querySelector("#start");
const stopButton = document.querySelector("#stop");
const languageInput = document.querySelector("#language");
const tokenInput = document.querySelector("#demo-token");
const statusOutput = document.querySelector("#status");
const partialOutput = document.querySelector("#partial");
const committedOutput = document.querySelector("#committed");
const outcomeOutput = document.querySelector("#outcome");
const latencyOutput = document.querySelector("#latency");

const turn = {
  phase: "idle",
  socket: null,
  stream: null,
  context: null,
  source: null,
  processor: null,
  sink: null,
  flushResolve: null,
  flushTimer: null,
  firstFrameSentAt: null,
  endSentAt: null,
};

function websocketUrl() {
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${window.location.host}/ws/voice-rag`;
}

function setStatus(message, kind = "") {
  statusOutput.textContent = message;
  statusOutput.dataset.kind = kind;
}

function resetResults() {
  partialOutput.textContent = "—";
  committedOutput.textContent = "—";
  outcomeOutput.textContent = "—";
  outcomeOutput.dataset.kind = "";
  latencyOutput.textContent = "Client timings appear after the final result.";
  turn.firstFrameSentAt = null;
  turn.endSentAt = null;
}

function setControls(active) {
  startButton.disabled = active;
  stopButton.disabled = !active;
  languageInput.disabled = active;
  tokenInput.disabled = active;
}

function restoreControls() {
  setControls(false);
  stopButton.disabled = true;
  tokenInput.value = "";
}

async function releaseMicrophone() {
  if (turn.flushTimer !== null) {
    window.clearTimeout(turn.flushTimer);
    turn.flushTimer = null;
  }
  turn.flushResolve = null;
  turn.stream?.getTracks().forEach((track) => track.stop());
  turn.source?.disconnect();
  turn.processor?.disconnect();
  turn.sink?.disconnect();
  if (turn.context && turn.context.state !== "closed") {
    try {
      await turn.context.close();
    } catch {
      // The browser can close an AudioContext during page teardown.
    }
  }
  turn.stream = null;
  turn.context = null;
  turn.source = null;
  turn.processor = null;
  turn.sink = null;
}

function closeSocket() {
  if (turn.socket && turn.socket.readyState < WebSocket.CLOSING) {
    turn.socket.close(1000, "Operator turn ended");
  }
  turn.socket = null;
}

function failTurn(message) {
  if (turn.phase === "failed" || turn.phase === "finished") {
    return;
  }
  turn.phase = "failed";
  setStatus(message, "error");
  closeSocket();
  void releaseMicrophone();
  restoreControls();
}

function formatMilliseconds(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} ms` : "not observed";
}

function showAnswer(payload) {
  const receivedAt = performance.now();
  const rag = payload?.rag;
  if (!rag || (rag.status !== "answered" && rag.status !== "refused")) {
    failTurn("The server returned an invalid final result.");
    return;
  }

  if (rag.status === "answered") {
    outcomeOutput.textContent = rag.answer ?? "No answer text was returned.";
    outcomeOutput.dataset.kind = "answered";
    setStatus("Grounded answer received.");
  } else {
    const reason = rag.refusal_reason ?? "unspecified";
    outcomeOutput.textContent = `Refused safely: ${reason}`;
    outcomeOutput.dataset.kind = "refused";
    setStatus("The grounded pipeline declined to answer.");
  }

  const clientFirstFrame = turn.firstFrameSentAt === null
    ? Number.NaN
    : receivedAt - turn.firstFrameSentAt;
  const clientPostEnd = turn.endSentAt === null
    ? Number.NaN
    : receivedAt - turn.endSentAt;
  const server = payload.latencies ?? {};
  latencyOutput.textContent = [
    `Client first-frame send() → answer receipt: ${formatMilliseconds(clientFirstFrame)}`,
    `Client end send() → answer receipt:        ${formatMilliseconds(clientPostEnd)}`,
    `Server audio EOF → answer:                 ${formatMilliseconds(server.audio_eof_to_answer_ms)}`,
    `Server first audio → answer:               ${formatMilliseconds(server.first_audio_to_answer_ms)}`,
  ].join("\n");

  turn.phase = "finished";
  void releaseMicrophone();
  restoreControls();
}

function handleServerMessage(message) {
  if (message.event === "partial_transcript") {
    partialOutput.textContent = message.text || "—";
    return;
  }
  if (message.event === "committed_transcript") {
    committedOutput.textContent = message.payload?.transcript || "—";
    return;
  }
  if (message.event === "answer") {
    showAnswer(message.payload);
    return;
  }
  if (message.error_code) {
    failTurn(message.message || "The voice pipeline refused this turn.");
  }
}

function connectForTurn(languageCode, token) {
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(websocketUrl());
    turn.socket = socket;
    let ready = false;
    const timeout = window.setTimeout(() => {
      if (!ready) {
        socket.close();
        reject(new Error("The voice server did not become ready in time."));
      }
    }, READY_TIMEOUT_MS);

    socket.addEventListener("open", () => {
      socket.send(JSON.stringify({
        event: "start",
        language_code: languageCode,
        demo_token: token,
      }));
      token = "";
      tokenInput.value = "";
    }, { once: true });

    socket.addEventListener("message", (event) => {
      if (typeof event.data !== "string") {
        return;
      }
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        if (!ready) {
          reject(new Error("The voice server returned an invalid message."));
        } else {
          failTurn("The voice server returned an invalid message.");
        }
        return;
      }

      if (message.event === "ready" && !ready) {
        if (
          message.audio_format !== "pcm_s16le"
          || message.sample_rate_hz !== TARGET_SAMPLE_RATE
          || message.commit_strategy !== "manual"
        ) {
          reject(new Error("The server requested an unsupported audio format."));
          return;
        }
        ready = true;
        window.clearTimeout(timeout);
        resolve();
        return;
      }
      handleServerMessage(message);
      if (!ready && message.error_code) {
        window.clearTimeout(timeout);
        reject(new Error(message.message || "Voice demo authentication failed."));
      }
    });

    socket.addEventListener("error", () => {
      if (!ready) {
        window.clearTimeout(timeout);
        reject(new Error("The voice WebSocket could not connect."));
      } else {
        failTurn("The voice WebSocket connection failed.");
      }
    });

    socket.addEventListener("close", () => {
      window.clearTimeout(timeout);
      if (!ready) {
        reject(new Error("The voice server closed before it was ready."));
      } else if (!["finished", "failed"].includes(turn.phase)) {
        failTurn("The voice server closed before returning a final result.");
      }
    });
  });
}

function handleWorkletMessage(event) {
  if (event.data?.event === "audio") {
    const buffer = event.data.buffer;
    if (!(buffer instanceof ArrayBuffer) || buffer.byteLength === 0) {
      return;
    }
    if (buffer.byteLength > MAX_FRAME_BYTES || buffer.byteLength % 2 !== 0) {
      failTurn("The browser produced an invalid PCM16 audio frame.");
      return;
    }
    if (
      ["recording", "stopping"].includes(turn.phase)
      && turn.socket?.readyState === WebSocket.OPEN
    ) {
      turn.socket.send(buffer);
      if (turn.firstFrameSentAt === null) {
        turn.firstFrameSentAt = performance.now();
      }
    }
    return;
  }
  if (event.data?.event === "stopped" && turn.flushResolve) {
    turn.flushResolve();
  }
}

async function beginMicrophone() {
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
    throw new Error("This browser does not support secure microphone capture.");
  }
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    throw new Error("This browser does not provide Web Audio.");
  }

  turn.stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      sampleRate: { ideal: TARGET_SAMPLE_RATE },
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
    video: false,
  });
  turn.context = new AudioContextClass({ latencyHint: "interactive" });
  await turn.context.audioWorklet.addModule("/static/pcm-worklet.js");
  turn.source = turn.context.createMediaStreamSource(turn.stream);
  turn.processor = new AudioWorkletNode(turn.context, "pcm16-resampler", {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    channelCount: 1,
    processorOptions: {
      outputSampleRate: TARGET_SAMPLE_RATE,
      batchSamples: BATCH_SAMPLES,
    },
  });
  turn.sink = turn.context.createGain();
  turn.sink.gain.value = 0;
  turn.processor.port.onmessage = handleWorkletMessage;
  turn.source.connect(turn.processor);
  turn.processor.connect(turn.sink);
  turn.sink.connect(turn.context.destination);
  await turn.context.resume();
  turn.processor.port.postMessage({ event: "start" });
}

async function startTurn() {
  if (turn.phase !== "idle" && !["failed", "finished"].includes(turn.phase)) {
    return;
  }
  const token = tokenInput.value;
  if (token.length < 16 || token.length > 512) {
    setStatus("Type the valid operator demo token before starting.", "error");
    tokenInput.focus();
    return;
  }

  resetResults();
  turn.phase = "connecting";
  setControls(true);
  stopButton.disabled = true;
  setStatus("Authorizing the same-origin voice connection…");

  try {
    await connectForTurn(languageInput.value, token);
    if (turn.phase === "failed") {
      return;
    }
    setStatus("Voice server ready. Requesting microphone access…");
    await beginMicrophone();
    if (turn.socket?.readyState !== WebSocket.OPEN) {
      throw new Error("The voice server closed before recording began.");
    }
    turn.phase = "recording";
    stopButton.disabled = false;
    setStatus("Recording. Speak now, then press Stop.");
  } catch (error) {
    failTurn(error instanceof Error ? error.message : "The voice turn could not start.");
  }
}

function flushWorklet() {
  if (!turn.processor) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    turn.flushResolve = () => {
      if (turn.flushTimer !== null) {
        window.clearTimeout(turn.flushTimer);
        turn.flushTimer = null;
      }
      turn.flushResolve = null;
      resolve();
    };
    turn.flushTimer = window.setTimeout(() => {
      turn.flushTimer = null;
      turn.flushResolve = null;
      resolve();
    }, FLUSH_TIMEOUT_MS);
    turn.processor.port.postMessage({ event: "stop" });
  });
}

async function stopTurn() {
  if (turn.phase !== "recording") {
    return;
  }
  turn.phase = "stopping";
  stopButton.disabled = true;
  setStatus("Flushing the final audio frame…");
  await flushWorklet();

  if (turn.firstFrameSentAt === null) {
    failTurn("No microphone audio was captured. Please try again.");
    return;
  }
  if (turn.socket?.readyState !== WebSocket.OPEN) {
    failTurn("The voice connection closed before Stop was sent.");
    return;
  }

  turn.socket.send(JSON.stringify({ event: "end" }));
  turn.endSentAt = performance.now();
  turn.phase = "waiting";
  setStatus("Transcript committed. Waiting for the grounded result…");
  await releaseMicrophone();
}

startButton.addEventListener("click", () => {
  void startTurn();
});
stopButton.addEventListener("click", () => {
  void stopTurn();
});
window.addEventListener("pagehide", () => {
  turn.phase = "failed";
  closeSocket();
  void releaseMicrophone();
});
