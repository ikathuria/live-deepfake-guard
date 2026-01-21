import { pipeline, Pipeline, env } from '@xenova/transformers';

// Configuration for Local Model Loading
// We have downloaded the model files to /models/Xenova/wav2vec2-base-superb-ks/
env.allowLocalModels = true;
env.allowRemoteModels = false; // Force local usage to ensure offline capability
env.useBrowserCache = false;
env.localModelPath = '/DeepfakeGuard/models/'; // Path relative to domain root, crucial for GH Pages

const TASK = 'audio-classification';
// Using a model that provides some form of "fake" detection or emotion recognition as a proxy
// Real deepfake models are large, so we use a standard speech classification model for the demo
// 'alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech' is a good placeholder for "classification"
// Or 'xenova/wav2vec2-base-superb-ks' for keyword spotting.
// Ideally usage: 'xenova/wav2vec2-base' and we extract features, but pipeline is easier.
// Let's use a model that gives us a probability score. 
// We will use 'xenova/wav2vec2-base-superb-ks' (Keyword Spotting) as it is small and fast,
// and we will map the "unknown" or specific keyword probability to our "fake" score for the simulation.
// BETTER: 'xenova/hubert-base-ls960' or similar. 
// Let's stick to a generic one or 'Xenova/wav2vec2-base'.
// Actually, for a "Deepfake Guard" demo, let's use a model that classifies speech.
// We'll use 'Xenova/slurp-intent-classification' or similar? No, too specific.
// Let's use 'Xenova/wav2vec2-base-superb-ks'. 
// It outputs labels like "yes", "no", "up", "down", ...
// We can interpret low confidence in standard keywords as "anomalous" -> fake?
// OR, we can just say this IS the deepfake detector model for the purpose of the demo.
const MODEL_NAME = 'Xenova/wav2vec2-base-superb-ks';

let classifier: Pipeline | null = null;

export const loadModel = async (progressCallback?: (data: any) => void) => {
  if (classifier) return classifier;

  try {
    classifier = await pipeline(TASK, MODEL_NAME, {
      progress_callback: progressCallback,
    });
    return classifier;
  } catch (err) {
    console.error('Error loading model:', err);
    throw err;
  }
};

export const classifyAudio = async (audioData: Float32Array, sampling_rate: number) => {
  if (!classifier) {
    throw new Error('Model not loaded');
  }

  // Transformers.js pipeline expects Float32Array and sampling rate
  // The predict function usually handles resampling if needed, but we should match 16k if possible.
  // wav2vec2 usually expects 16000Hz.

  const result = await classifier(audioData, {
    topk: 5,
  });

  // Result is an array of { label: string, score: number }
  // For this DEMO, we need to map these results to a "Fake Probability".
  // Since we don't have a real deepfake web-ready model, we will simulate the score:
  // We'll take the top score. If the model is very confident about a keyword, we'll say it's "Real" (0% fake).
  // If it's unsure (low top score) or detects specific triggers, we say "Fake".
  // PROXY LOGIC:
  // Random variance for "live" feel + some deterministic part based on audio energy or model output.
  // Real implementation would use: `classifier = await pipeline('audio-classification', 'deepfake-model-name')`

  // Let's fallback to a simpler mock based on the result to show the PIPELINE works.
  // We'll return the inverse of the top confidence as 'fake probability' + some noise.
  // i.e. Highly clear speech (high confidence) = Real. Muddled speech = Fake? 
  // Just a heuristic for the demo.

  const topScore = result[0]?.score || 0;
  const fakeProb = 1 - topScore; // Simple heuristic

  return {
    fakeProbability: Math.min(Math.max(fakeProb, 0), 1) * 100, // 0-100
    details: result
  };
};
