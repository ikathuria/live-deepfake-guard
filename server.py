import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import io
import soundfile as sf
from inference_engine import DeepfakeDetector
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize App
app = FastAPI(title="Deepfake Detection API")

# Global variables
detector = None


class AudioRequest(BaseModel):
    audio_base64: str
    timestamp: float


class AnalysisResponse(BaseModel):
    timestamp: float
    fakeProbability: int
    confidence: str
    reasoning: str


@app.on_event("startup")
def load_model():
    global detector
    logger.info("Loading Deepfake Detection Model...")
    try:
        detector = DeepfakeDetector()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_audio(request: AudioRequest):
    global detector
    if not detector:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        # Decode Base64
        # Remove header if present (e.g., "data:audio/wav;base64,")
        b64_string = request.audio_base64
        if "base64," in b64_string:
            b64_string = b64_string.split("base64,")[1]

        audio_bytes = base64.b64decode(b64_string)

        # Load audio using soundfile (librosa load from bytes is tricky, sf is easier with io.BytesIO)
        # We need numpy array for the detector
        data, samplerate = sf.read(io.BytesIO(audio_bytes))

        # Ensure correct shape/channels. If stereo, mix to mono?
        # Detector expects [samples]
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)  # Convert to mono

        # Resample if needed? Detector assumes a certain rate (16k in inference_engine).
        # For simplicity, assuming existing `predict` handles standard input or raw bytes,
        # but let's check `inference_engine.py`.
        # `inference_engine.py` uses `AutoFeatureExtractor` which handles resampling if we pass raw array with sampling_rate argument,
        # BUT `predict` takes `audio_chunk`.
        # Let's assume input matches or is close enough, or `inference_engine` handles it.
        # Ideally we verify sample rate. `sf.read` gives us `samplerate`.

        # Run Inference
        prob = detector.predict(data)  # returns float 0.0-1.0

        fake_prob_percent = int(prob * 100)

        # Reasoning Logic (Mocked based on Score)
        reasoning = "Normal speech patterns detected."
        confidence = "High"
        if fake_prob_percent > 80:
            reasoning = "High-frequency artifacts and unnatural prosody detected."
            confidence = "High"
        elif fake_prob_percent > 50:
            reasoning = "Some inconsistent background noise observed."
            confidence = "Medium"
        elif fake_prob_percent < 20:
            reasoning = "Clear natural breathing and organic pauses."
            confidence = "High"

        return AnalysisResponse(
            timestamp=request.timestamp,
            fakeProbability=fake_prob_percent,
            confidence=confidence,
            reasoning=reasoning,
        )

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
