# import torch
# import numpy as np
# from transformers import Wav2Vec2Model, Wav2Vec2Processor

# MODEL_NAME = "facebook/wav2vec2-xls-r-300m"


# class Wav2VecXLSR:
#     def __init__(self):
#         self.processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
#         self.model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
#         self.model.eval()

#     def extract_features(self, waveform, sample_rate=16000):
#         # ✅ Ensure waveform is 1D numpy array
#         if isinstance(waveform, torch.Tensor):
#             waveform = waveform.squeeze().cpu().numpy()
#         else:
#             waveform = np.asarray(waveform).squeeze()

#         # Safety check
#         assert waveform.ndim == 1, f"Expected 1D waveform, got {waveform.shape}"

#         inputs = self.processor(
#             waveform,
#             sampling_rate=sample_rate,
#             return_tensors="pt",
#             padding=True
#         )

#         with torch.no_grad():
#             outputs = self.model(**inputs)

#         # (batch, time, 1024)
#         return outputs.last_hidden_state

# import torch
# import numpy as np
# from transformers import AutoFeatureExtractor, Wav2Vec2Model

# MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

# class Wav2VecXLSR:
#     def __init__(self):
#         # ✅ Use AutoFeatureExtractor instead of Processor to avoid Tokenizer error
#         self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
#         self.model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
#         self.model.eval()

#     def extract_features(self, waveform, sample_rate=16000):
#         # ✅ Handle input types
#         if isinstance(waveform, torch.Tensor):
#             waveform = waveform.squeeze().cpu().numpy()
#         else:
#             waveform = np.asarray(waveform).squeeze()

#         # Ensure 1D
#         if waveform.ndim > 1:
#             waveform = waveform.flatten()

#         # ✅ Correct way to use Feature Extractor
#         inputs = self.feature_extractor(
#             waveform,
#             sampling_rate=sample_rate,
#             return_tensors="pt",
#             padding=True
#         ).to(self.device)

#         with torch.no_grad():
#             outputs = self.model(**inputs)

#         # Returns (batch, time, 1024)
#         return outputs.last_hidden_state



import torch
import numpy as np
from transformers import AutoFeatureExtractor, Wav2Vec2Model

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"


class Wav2VecXLSR:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Feature extractor (NO tokenizer)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

        # XLS-R backbone
        self.model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def extract_features(self, waveform, sample_rate: int = 16000):
        """
        Input:
            waveform: np.ndarray or torch.Tensor (1D)
        Output:
            torch.Tensor -> (B, T, 1024)
        """

        # ---- Normalize input ----
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()

        waveform = np.asarray(waveform, dtype=np.float32).squeeze()

        if waveform.ndim != 1:
            waveform = waveform.flatten()

        # ---- Feature extraction ----
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=False
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # ---- Forward ----
        with torch.no_grad():
            outputs = self.model(**inputs)

        # (B, T, 1024)
        return outputs.last_hidden_state
