# import torch
# import numpy as np

# from app.models.wav2vec2_xlsr import Wav2VecXLSR
# from app.models.aasist3 import AASIST

# # -------------------- Globals --------------------
# wav2vec = None
# aasist = None
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # -------------------- Utils --------------------
# def apply_pre_emphasis(y, coef=0.97):
#     """
#     Optional: enhances high-frequency artifacts.
#     Safe for short audio.
#     """
#     if y is None or len(y) < 2:
#         return y
#     return np.append(y[0], y[1:] - coef * y[:-1])


# # -------------------- Model Loader --------------------
# def load_models():
#     global wav2vec, aasist

#     if wav2vec is None:
#         wav2vec = Wav2VecXLSR()
#         # Move internal model to device if supported
#         if hasattr(wav2vec, "to"):
#             wav2vec.to(device)

#     if aasist is None:
#         aasist = AASIST()
#         aasist.to(device)
#         aasist.eval()


# # -------------------- Main Inference --------------------
# def detect_ai_voice(waveform, sample_rate: int = 16000):
#     """
#     Args:
#         waveform: np.ndarray or torch.Tensor (mono)
#         sample_rate: expected 16000
#     Returns:
#         (label, confidence)
#     """
#     load_models()

#     # ---- 1. Input normalization ----
#     if isinstance(waveform, torch.Tensor):
#         waveform = waveform.detach().cpu().numpy()

#     waveform = waveform.flatten().astype(np.float32)

#     # Optional (can disable if needed)
#     waveform = apply_pre_emphasis(waveform)

#     # ---- 2. XLS-R feature extraction ----
#     # extract_features MUST return torch.Tensor [B, T, C] or [B, C, T]
#     features = wav2vec.extract_features(waveform, sample_rate)

#     if not isinstance(features, torch.Tensor):
#         raise RuntimeError("Wav2VecXLSR.extract_features must return a torch.Tensor")

#     features = features.to(device)

#     # ---- 3. AASIST inference ----
#     with torch.no_grad():
#         logits = aasist(features)

#     probs = torch.softmax(logits, dim=-1)

#     # Assumption: index 0 = human, 1 = AI
#     human_prob = probs[0][0].item()
#     ai_prob = probs[0][1].item()

#     if ai_prob > human_prob:
#         return "AI_GENERATED", round(ai_prob, 4)
#     else:
#         return "HUMAN", round(human_prob, 4)



# import torch
# import numpy as np
# import os
# from huggingface_hub import login

# from app.models.wav2vec2_xlsr import Wav2VecXLSR
# from app.models.aasist3 import AASIST

# # -------------------- Globals --------------------
# wav2vec = None
# aasist = None
# # Force CPU on Render Free Tier to avoid CUDA initialization memory overhead
# device = torch.device("cpu") 

# # -------------------- Utils --------------------
# def apply_pre_emphasis(y, coef=0.97):
#     if y is None or len(y) < 2:
#         return y
#     return np.append(y[0], y[1:] - coef * y[:-1])

# # -------------------- Model Loader --------------------
# def load_models():
#     global wav2vec, aasist

#     # 1. Login to HF using the token from Render Environment Variables
#     hf_token = os.getenv("HF_TOKEN")
#     if hf_token:
#         try:
#             login(token=hf_token)
#             print("Successfully authenticated with Hugging Face.")
#         except Exception as e:
#             print(f"HF Login failed: {e}")

#     # 2. Load XLS-R
#     if wav2vec is None:
#         print("Initializing Wav2VecXLSR... (This may take a moment)")
#         wav2vec = Wav2VecXLSR()
#         if hasattr(wav2vec, "to"):
#             wav2vec.to(device)

#     # 3. Load AASIST
#     if aasist is None:
#         print("Initializing AASIST3...")
#         aasist = AASIST()
#         aasist.to(device)
#         aasist.eval()
    
#     print("All models loaded successfully.")

# # -------------------- Main Inference --------------------
# def detect_ai_voice(waveform, sample_rate: int = 16000):
#     # CRITICAL: If models aren't loaded, load them now
#     if wav2vec is None or aasist is None:
#         load_models()

#     # ---- 1. Input normalization ----
#     if isinstance(waveform, torch.Tensor):
#         waveform = waveform.detach().cpu().numpy()

#     waveform = waveform.flatten().astype(np.float32)
#     waveform = apply_pre_emphasis(waveform)

#     # ---- 2. XLS-R feature extraction ----
#     features = wav2vec.extract_features(waveform, sample_rate)

#     if not isinstance(features, torch.Tensor):
#         raise RuntimeError("Wav2VecXLSR.extract_features must return a torch.Tensor")

#     features = features.to(device)

#     # ---- 3. AASIST inference ----
#     with torch.no_grad():
#         logits = aasist(features)

#     probs = torch.softmax(logits, dim=-1)

#     # Assumption: index 0 = human, 1 = AI
#     # Adding a safety check for tensor shape
#     if probs.ndim == 1:
#         human_prob = probs[0].item()
#         ai_prob = probs[1].item()
#     else:
#         human_prob = probs[0][0].item()
#         ai_prob = probs[0][1].item()

#     if ai_prob > human_prob:
#         return "AI_GENERATED", round(ai_prob, 4)
#     else:
#         return "HUMAN", round(human_prob, 4)



# inference.py
# import os
# import gc
# import torch
# import numpy as np
# from huggingface_hub import login

# # -------------------- Utils --------------------
# def apply_pre_emphasis(y, coef=0.97):
#     """
#     Apply pre-emphasis to audio waveform
#     """
#     if y is None or len(y) < 2:
#         return y
#     return np.append(y[0], y[1:] - coef * y[:-1])

# # -------------------- Model Containers --------------------
# # Use globals for lazy-loading and caching
# wav2vec_model = None
# aasist_model = None
# device = torch.device("cpu")  # Force CPU

# # -------------------- Main Inference --------------------
# def detect_ai_voice(waveform, sample_rate: int = 16000):
#     """
#     Detect whether voice is AI-generated or human.
#     Returns: ("AI_GENERATED" or "HUMAN", confidence_score)
#     """

#     global wav2vec_model, aasist_model

#     # 1️⃣ Lazy-load models on first request
#     if wav2vec_model is None or aasist_model is None:
#         try:
#             from app.models.wav2vec2_xlsr import Wav2VecXLSR
#             from app.models.aasist3 import AASIST

#             # HuggingFace login if token exists
#             hf_token = os.getenv("HF_TOKEN")
#             if hf_token:
#                 login(token=hf_token)

#             # Initialize models
#             wav2vec_model = Wav2VecXLSR()   # no .to() needed
#             aasist_model = AASIST()          # no .to() needed

#         except Exception as e:
#             raise RuntimeError(f"Failed to load models: {e}")

#     try:
#         # 2️⃣ Preprocess waveform
#         if isinstance(waveform, torch.Tensor):
#             waveform = waveform.detach().cpu().numpy()
#         waveform = waveform.flatten().astype(np.float32)
#         waveform = apply_pre_emphasis(waveform)

#         # 3️⃣ Extract features with wav2vec
#         with torch.inference_mode():
#             features = wav2vec_model.extract_features(torch.tensor(waveform, dtype=torch.float32),
#                                                        sample_rate)

#         # 4️⃣ Run classification with AASIST
#         features_tensor = features if isinstance(features, torch.Tensor) else torch.tensor(features)
#         with torch.inference_mode():
#             logits = aasist_model(features_tensor)
#             probs = torch.softmax(logits, dim=-1)

#         # 5️⃣ Get human vs AI probability
#         if probs.ndim == 1:
#             human_prob, ai_prob = probs[0].item(), probs[1].item()
#         else:
#             human_prob, ai_prob = probs[0][0].item(), probs[0][1].item()

#         # 6️⃣ Clean up temporary tensors
#         gc.collect()

#         # 7️⃣ Return result
#         if ai_prob > human_prob:
#             return "AI_GENERATED", round(ai_prob, 4)
#         return "HUMAN", round(human_prob, 4)

#     except Exception as e:
#         gc.collect()
#         raise RuntimeError(f"Inference failed: {e}")


# import os
# import gc
# import torch
# import numpy as np
# from huggingface_hub import login
# from app.models.wav2vec_base import Wav2VecXLSR


# # -------------------- Utils --------------------
# def apply_pre_emphasis(y, coef=0.97):
#     if y is None or len(y) < 2:
#         return y
#     return np.append(y[0], y[1:] - coef * y[:-1])

# # -------------------- Main Inference --------------------
# def detect_ai_voice(waveform, sample_rate: int = 16000):
#     # 1️⃣ Setup Environment
#     hf_token = os.getenv("HF_TOKEN")
#     if hf_token:
#         login(token=hf_token)

#     try:
#         # 2️⃣ Extract Features (Wav2Vec)
#         from app.models.wav2vec2_xlsr import Wav2VecXLSR
        
#         # Preprocess waveform first to save memory
#         if isinstance(waveform, torch.Tensor):
#             waveform = waveform.detach().cpu().numpy()
#         waveform = waveform.flatten().astype(np.float32)
#         waveform = apply_pre_emphasis(waveform)
        
#         # Load, Use, and Delete Wav2Vec immediately
#         wav2vec = Wav2VecXLSR()
#         with torch.inference_mode():
#             # Ensure input is a tensor
#             features = wav2vec.extract_features(torch.from_numpy(waveform), sample_rate)
        
#         del wav2vec  # Free ~300MB immediately
#         gc.collect()

#         # 3️⃣ Classify (AASIST)
#         from app.models.aasist3 import AASIST
#         aasist = AASIST()
        
#         with torch.inference_mode():
#             logits = aasist(features)
#             probs = torch.softmax(logits, dim=-1)
        
#         del aasist   # Free AASIST RAM
        
#         # 4️⃣ Result Logic
#         if probs.ndim == 1:
#             human_prob, ai_prob = probs[0].item(), probs[1].item()
#         else:
#             human_prob, ai_prob = probs[0][0].item(), probs[0][1].item()

#         gc.collect() # Final cleanup
        
#         if ai_prob > human_prob:
#             return "AI_GENERATED", round(ai_prob, 4)
#         return "HUMAN", round(human_prob, 4)

#     except Exception as e:
#         gc.collect()
#         raise RuntimeError(f"Inference failed: {str(e)}")


import os
import gc
import torch
import numpy as np
from huggingface_hub import login
from app.models.aasist3_small import AASIST


# -------------------- Utils --------------------
def apply_pre_emphasis(y, coef=0.97):
    """Apply pre-emphasis filter to audio"""
    if y is None or len(y) < 2:
        return y
    return np.append(y[0], y[1:] - coef * y[:-1])

# -------------------- Lazy-load models --------------------
wav2vec_model = None
aasist_model = None

# -------------------- Main Inference --------------------
def detect_ai_voice(waveform, sample_rate: int = 16000):
    global wav2vec_model, aasist_model

    # 1️⃣ HuggingFace login if token exists
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # 2️⃣ Lazy-load models
    if wav2vec_model is None:
        from transformers import AutoFeatureExtractor, Wav2Vec2Model

        class Wav2VecBase:
            """Wrapper for smaller Wav2Vec2-base-960h model"""
            def __init__(self):
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                    "facebook/wav2vec2-base-960h"
                )
                self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
                self.model.eval()

            def extract_features(self, waveform, sample_rate=16000):
                if not isinstance(waveform, torch.Tensor):
                    waveform = torch.tensor(waveform, dtype=torch.float32)
                inputs = self.feature_extractor(
                    waveform,
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=False
                )
                with torch.inference_mode():
                    outputs = self.model(**inputs)
                return outputs.last_hidden_state  # (B, T, 768)

        wav2vec_model = Wav2VecBase()

    if aasist_model is None:
        # Import adapted AASIST (first layer modified for 768 features)
        from app.models.aasist3_small import AASIST  # your adapted version
        aasist_model = AASIST()
        aasist_model.eval()

    try:
        # 3️⃣ Preprocess waveform
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        waveform = waveform.flatten().astype(np.float32)
        waveform = apply_pre_emphasis(waveform)

        # 4️⃣ Extract features
        with torch.inference_mode():
            features = wav2vec_model.extract_features(torch.from_numpy(waveform), sample_rate)

        # 5️⃣ Classify with adapted AASIST
        with torch.inference_mode():
            logits = aasist_model(features)
            probs = torch.softmax(logits, dim=-1)

        # 6️⃣ Get probabilities
        if probs.ndim == 1:
            human_prob, ai_prob = probs[0].item(), probs[1].item()
        else:
            human_prob, ai_prob = probs[0][0].item(), probs[0][1].item()

        gc.collect()  # final cleanup

        return ("AI_GENERATED", round(ai_prob, 4)) if ai_prob > human_prob else ("HUMAN", round(human_prob, 4))

    except Exception as e:
        gc.collect()
        raise RuntimeError(f"Inference failed: {str(e)}")
