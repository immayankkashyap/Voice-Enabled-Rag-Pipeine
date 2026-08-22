"use strict";

const OUTPUT_SAMPLE_RATE = 16_000;
const BATCH_SAMPLES = 320; // 20 ms / 640 bytes, safely below the 100 ms limit.
const MAX_BATCH_SAMPLES = 1_600; // 100 ms of mono PCM16 at 16 kHz.

class PCM16Resampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const requestedRate = options.processorOptions?.outputSampleRate;
    const requestedBatch = options.processorOptions?.batchSamples;
    this.outputRate = requestedRate === OUTPUT_SAMPLE_RATE
      ? requestedRate
      : OUTPUT_SAMPLE_RATE;
    this.batchSamples = Number.isInteger(requestedBatch)
      ? Math.min(Math.max(requestedBatch, 1), MAX_BATCH_SAMPLES)
      : BATCH_SAMPLES;
    this.inputPerOutput = sampleRate / this.outputRate;
    this.recording = false;
    this.inputSamples = [];
    this.outputSamples = [];
    this.readPosition = 0;

    this.port.onmessage = (event) => {
      if (event.data?.event === "start") {
        this.inputSamples = [];
        this.outputSamples = [];
        this.readPosition = 0;
        this.recording = true;
      } else if (event.data?.event === "stop") {
        this.recording = false;
        this.resample(true);
        this.emit(true);
        this.port.postMessage({ event: "stopped" });
      }
    };
  }

  appendMono(input) {
    const frames = input[0]?.length ?? 0;
    for (let frame = 0; frame < frames; frame += 1) {
      let sample = 0;
      for (let channel = 0; channel < input.length; channel += 1) {
        sample += input[channel][frame] ?? 0;
      }
      this.inputSamples.push(sample / input.length);
    }
  }

  resample(flush) {
    const boundary = flush
      ? this.inputSamples.length
      : Math.max(0, this.inputSamples.length - 1);
    while (this.readPosition < boundary) {
      const leftIndex = Math.floor(this.readPosition);
      const rightIndex = Math.min(leftIndex + 1, this.inputSamples.length - 1);
      const fraction = this.readPosition - leftIndex;
      const left = this.inputSamples[leftIndex] ?? 0;
      const right = this.inputSamples[rightIndex] ?? left;
      this.outputSamples.push(left + (right - left) * fraction);
      this.readPosition += this.inputPerOutput;
    }

    if (this.inputSamples.length > 1) {
      const discard = Math.min(
        Math.floor(this.readPosition),
        this.inputSamples.length - 1,
      );
      if (discard > 0) {
        this.inputSamples.splice(0, discard);
        this.readPosition -= discard;
      }
    }
  }

  emit(flush) {
    while (
      this.outputSamples.length >= this.batchSamples
      || (flush && this.outputSamples.length > 0)
    ) {
      const count = flush
        ? Math.min(this.batchSamples, this.outputSamples.length)
        : this.batchSamples;
      const pcm = new Int16Array(count);
      for (let index = 0; index < count; index += 1) {
        const sample = Math.max(-1, Math.min(1, this.outputSamples[index]));
        pcm[index] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      }
      this.outputSamples.splice(0, count);
      this.port.postMessage(
        { event: "audio", buffer: pcm.buffer },
        [pcm.buffer],
      );
    }
  }

  process(inputs) {
    if (!this.recording) {
      return true;
    }
    const input = inputs[0];
    if (input?.length) {
      this.appendMono(input);
      this.resample(false);
      this.emit(false);
    }
    return true;
  }
}

registerProcessor("pcm16-resampler", PCM16Resampler);
