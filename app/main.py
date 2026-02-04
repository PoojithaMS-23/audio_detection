# from fastapi import FastAPI, HTTPException, Header
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import base64
# import os

# from app.utils.audio_utils import load_audio_from_bytes
# from app.models.inference import detect_ai_voice


# app = FastAPI(title="AI Voice Detection API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Testing only
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # API Key (use env variable in production)
# API_KEY = os.getenv("API_KEY", "sk_test_123456789")

# SUPPORTED_LANGUAGES = [
#     "Tamil",
#     "English",
#     "Hindi",
#     "Malayalam",
#     "Telugu"
# ]


# class AudioRequest(BaseModel):
#     language: str
#     audioFormat: str
#     audioBase64: str


# @app.post("/api/voice-detection")
# def detect_voice(
#     request: AudioRequest,
#     x_api_key: str = Header(None)
# ):
#     # 1. API Key check
#     if x_api_key != API_KEY:
#         raise HTTPException(status_code=401, detail="Invalid API Key")

#     # 2. Language check
#     if request.language not in SUPPORTED_LANGUAGES:
#         raise HTTPException(status_code=400, detail="Unsupported language")

#     # 3. Base64 decode
#     try:
#         audio_bytes = base64.b64decode(request.audioBase64)
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid Base64 audio")

#     # 4. Load & preprocess audio
#     try:
#         y, sr = load_audio_from_bytes(audio_bytes)
#     except Exception as e:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Audio processing failed: {str(e)}"
#         )

#     # 5. AI vs Human detection
#     classification, confidence = detect_ai_voice(y, sr)

#     explanation = (
#         "Speech patterns indicate synthetic generation"
#         if classification == "AI_GENERATED"
#         else "Speech patterns are consistent with human voice"
#     )

#     return {
#         "status": "success",
#         "language": request.language,
#         "classification": classification,
#         "confidenceScore": confidence,
#         "explanation": explanation
#     }


# @app.get("/")
# def root():
#     return {"status": "API is running"}




# import os
# import base64
# from fastapi import FastAPI, HTTPException, Header
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv
# from app.utils.audio_utils import load_audio_from_bytes
# from app.models.inference import detect_ai_voice

# app = FastAPI(title="AI Voice Detection API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # For testing; restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# load_dotenv()

# # ✅ Load API key from environment
# API_KEY = os.getenv("API_KEY")
# if not API_KEY:
#     raise RuntimeError("API_KEY environment variable not set!")

# SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

# class AudioRequest(BaseModel):
#     language: str
#     audioFormat: str
#     audioBase64: str

# @app.get("/")
# def health_check():
#     return {"status": "alive"}

# @app.head("/")
# def health_check_head():
#     return {"status": "alive"}

# @app.post("/api/voice-detection")
# def detect_voice(request: AudioRequest, x_api_key: str = Header(None)):

#     # 1️⃣ API Key validation
#     if x_api_key != API_KEY:
#         raise HTTPException(status_code=401, detail={"status": "error", "message": "Invalid API key or malformed request"})

#     # 2️⃣ Language check
#     if request.language not in SUPPORTED_LANGUAGES:
#         raise HTTPException(status_code=400, detail={"status": "error", "message": "Unsupported language"})

#     # 3️⃣ Base64 decode
#     try:
#         audio_bytes = base64.b64decode(request.audioBase64)
#     except Exception:
#         raise HTTPException(status_code=400, detail={"status": "error", "message": "Invalid Base64 audio"})

#     # 4️⃣ Load audio
#     try:
#         y, sr = load_audio_from_bytes(audio_bytes)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail={"status": "error", "message": f"Audio processing failed: {str(e)}"})

#     # 5️⃣ Detect AI vs Human
#     classification, confidence = detect_ai_voice(y, sr)

#     explanation = (
#         "Speech patterns indicate synthetic generation"
#         if classification == "AI_GENERATED"
#         else "Speech patterns are consistent with human voice"
#     )

#     return {
#         "status": "success",
#         "language": request.language,
#         "classification": classification,
#         "confidenceScore": confidence,
#         "explanation": explanation
#     }

# # @app.get("/")
# # def root():
# #     return {"status": "API is running"}


import os
import base64
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from app.utils.audio_utils import load_audio_from_bytes
from app.models.inference import detect_ai_voice, load_models

app = FastAPI(title="AI Voice Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise RuntimeError("API_KEY environment variable not set!")

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

class AudioRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

# --- Health Checks for Render ---
@app.get("/")
@app.head("/")
def health_check():
    return {"status": "alive", "message": "Voice Detection API is running"}

# --- Warmup Endpoint ---
@app.post("/api/warmup")
def warmup(x_api_key: str = Header(None, alias="x-api-key")):
    """Triggers model loading without needing audio data."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    load_models() # This triggers the Lazy Load
    return {"status": "success", "message": "Models are warmed up and ready"}

@app.post("/api/voice-detection")
def detect_voice(request: AudioRequest, x_api_key: str = Header(None, alias="x-api-key")):

    # 1️⃣ API Key validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail={"status": "error", "message": "Invalid API key"})

    # 2️⃣ Language check
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail={"status": "error", "message": f"Unsupported language: {request.language}"})

    # 3️⃣ Base64 decode
    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail={"status": "error", "message": "Invalid Base64 audio"})

    # 4️⃣ Load audio
    try:
        y, sr = load_audio_from_bytes(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"status": "error", "message": f"Audio processing failed: {str(e)}"})

    # 5️⃣ Detect AI vs Human (Lazy Loading happens here if not warmed up)
    try:
        classification, confidence = detect_ai_voice(y, sr)
    except Exception as e:
        # Catch internal model errors so the whole server doesn't 502
        raise HTTPException(status_code=500, detail={"status": "error", "message": f"Inference failed: {str(e)}"})

    explanation = (
        "Speech patterns indicate synthetic generation"
        if classification == "AI_GENERATED"
        else "Speech patterns are consistent with human voice"
    )

    return {
        "status": "success",
        "language": request.language,
        "classification": classification,
        "confidenceScore": confidence,
        "explanation": explanation
    }