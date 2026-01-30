from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import io
import librosa
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for testing, allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API key from environment
API_KEY = os.getenv("API_KEY", "sk_test_123456789")  # fallback for testing

# Supported languages
SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

class AudioRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.post("/api/voice-detection")
def detect_voice(
    request: AudioRequest,
    x_api_key: str = Header(None)
):
    # 1️⃣ Check API key
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # 2️⃣ Check language
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    # 3️⃣ Decode Base64 to bytes
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # 4️⃣ Load audio with librosa
    try:
        audio_file = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_file, sr=None)  # sr=None to preserve original sampling rate
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading audio: {str(e)}")

    # 5️⃣ Dummy AI analysis (replace with your model)
    # For demo purposes, we randomly return HUMAN/AI_GENERATED
    import random
    classification = random.choice(["HUMAN", "AI_GENERATED"])
    confidence = round(random.uniform(0.7, 0.99), 2)
    explanation = "Detected features match expected patterns for demo"

    # 6️⃣ Return JSON response
    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }
