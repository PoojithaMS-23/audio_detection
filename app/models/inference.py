# import torch
# from app.models.wav2vec2_xlsr import Wav2VecXLSR
# from app.models.aasist3 import AASIST3Classifier

# wav2vec = Wav2VecXLSR()
# aasist = AASIST3Classifier()

# aasist.eval()

# def detect_ai_voice(audio, sr=16000):
#     features = wav2vec.extract_features(audio, sr)

#     with torch.no_grad():
#         logits = aasist(features)
#         probs = torch.softmax(logits, dim=1)

#     ai_score = probs[0][1].item()
#     human_score = probs[0][0].item()

#     if ai_score > human_score:
#         return "AI_GENERATED", round(ai_score, 2)
#     else:
#         return "HUMAN", round(human_score, 2)

# app/models/inference.py


# import torch
# from app.models.wav2vec2_xlsr import Wav2VecXLSR
# from app.models.aasist3 import AASIST

# # Lazy-loaded models (IMPORTANT)
# wav2vec = None
# aasist = None
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def load_models():
#     global wav2vec, aasist

#     if wav2vec is None:
#         wav2vec = Wav2VecXLSR()

#     if aasist is None:
#         aasist = AASIST()
#         aasist.to(device)
#         aasist.eval()


# def detect_ai_voice(waveform, sample_rate=16000):
#     """
#     Returns:
#         classification: "AI_GENERATED" | "HUMAN"
#         confidence: float (0–1)
#     """

#     load_models()

#     # Ensure torch tensor
#     if not isinstance(waveform, torch.Tensor):
#         waveform = torch.tensor(waveform).unsqueeze(0)

#     waveform = waveform.to(device)

#     # 1️⃣ Extract XLS-R embeddings
#     features = wav2vec.extract_features(waveform, sample_rate)
#     features = features.to(device)

#     # 2️⃣ AASIST inference
#     with torch.no_grad():
#         logits = aasist(features)

#     probs = torch.softmax(logits, dim=-1)

#     human_prob = probs[0][0].item()
#     ai_prob = probs[0][1].item()

#     if ai_prob > human_prob:
#         return "AI_GENERATED", round(ai_prob, 4)
#     else:
#         return "HUMAN", round(human_prob, 4)



# import torch
# import numpy as np
# from app.models.wav2vec2_xlsr import Wav2VecXLSR
# # Assuming your AASIST class is defined in aasist3.py
# from app.models.aasist3 import AASIST 

# wav2vec = None
# aasist = None
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def apply_pre_emphasis(y, coef=0.97):
#     """Secret sauce: sharpens high-frequency AI artifacts."""
#     return np.append(y[0], y[1:] - coef * y[:-1])

# def load_models():
#     global wav2vec, aasist
#     if wav2vec is None:
#         wav2vec = Wav2VecXLSR()
#     if aasist is None:
#         # If using MTUCI/AASIST3, it might need trust_remote_code=True
#         aasist = AASIST() 
#         aasist.to(device)
#         aasist.eval()

# def detect_ai_voice(waveform, sample_rate=16000):
#     load_models()

#     # 1️⃣ Pre-processing
#     if isinstance(waveform, torch.Tensor):
#         waveform = waveform.cpu().numpy()
    
#     waveform = waveform.flatten()
#     waveform = apply_pre_emphasis(waveform)

#     # 2️⃣ Extract XLS-R embeddings
#     # wav2vec.extract_features now handles the tensor conversion internally
#     features = wav2vec.extract_features(waveform, sample_rate)
    
#     # 3️⃣ AASIST inference
#     with torch.no_grad():
#         # Ensure features are on the correct device
#         features = features.to(device)
#         logits = aasist(features)

#     probs = torch.softmax(logits, dim=-1)

#     # Note: AASIST output indices vary by model, 
#     # usually [0] is real, [1] is fake
#     human_prob = probs[0][0].item()
#     ai_prob = probs[0][1].item()

#     if ai_prob > human_prob:
#         return "AI_GENERATED", round(ai_prob, 4)
#     else:
#         return "HUMAN", round(human_prob, 4)


import torch
import numpy as np

from app.models.wav2vec2_xlsr import Wav2VecXLSR
from app.models.aasist3 import AASIST

# -------------------- Globals --------------------
wav2vec = None
aasist = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Utils --------------------
def apply_pre_emphasis(y, coef=0.97):
    """
    Optional: enhances high-frequency artifacts.
    Safe for short audio.
    """
    if y is None or len(y) < 2:
        return y
    return np.append(y[0], y[1:] - coef * y[:-1])


# -------------------- Model Loader --------------------
def load_models():
    global wav2vec, aasist

    if wav2vec is None:
        wav2vec = Wav2VecXLSR()
        # Move internal model to device if supported
        if hasattr(wav2vec, "to"):
            wav2vec.to(device)

    if aasist is None:
        aasist = AASIST()
        aasist.to(device)
        aasist.eval()


# -------------------- Main Inference --------------------
def detect_ai_voice(waveform, sample_rate: int = 16000):
    """
    Args:
        waveform: np.ndarray or torch.Tensor (mono)
        sample_rate: expected 16000
    Returns:
        (label, confidence)
    """
    load_models()

    # ---- 1. Input normalization ----
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()

    waveform = waveform.flatten().astype(np.float32)

    # Optional (can disable if needed)
    waveform = apply_pre_emphasis(waveform)

    # ---- 2. XLS-R feature extraction ----
    # extract_features MUST return torch.Tensor [B, T, C] or [B, C, T]
    features = wav2vec.extract_features(waveform, sample_rate)

    if not isinstance(features, torch.Tensor):
        raise RuntimeError("Wav2VecXLSR.extract_features must return a torch.Tensor")

    features = features.to(device)

    # ---- 3. AASIST inference ----
    with torch.no_grad():
        logits = aasist(features)

    probs = torch.softmax(logits, dim=-1)

    # Assumption: index 0 = human, 1 = AI
    human_prob = probs[0][0].item()
    ai_prob = probs[0][1].item()

    if ai_prob > human_prob:
        return "AI_GENERATED", round(ai_prob, 4)
    else:
        return "HUMAN", round(human_prob, 4)
