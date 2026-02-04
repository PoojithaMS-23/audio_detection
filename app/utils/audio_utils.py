# # app/utils/audio_utils.py

# import io
# import base64
# import torch
# import torchaudio

# TARGET_SAMPLE_RATE = 16000


# def load_audio_from_bytes(audio_base64: str) -> torch.Tensor:
#     """
#     Base64 audio → mono waveform tensor at 16kHz
#     Shape: (1, T)
#     """

#     # Decode base64
#     audio_bytes = base64.b64decode(audio_base64)
#     buffer = io.BytesIO(audio_bytes)

#     # Load audio
#     waveform, sample_rate = torchaudio.load(buffer)

#     # Convert to mono
#     if waveform.shape[0] > 1:
#         waveform = waveform.mean(dim=0, keepdim=True)

#     # Resample
#     if sample_rate != TARGET_SAMPLE_RATE:
#         resampler = torchaudio.transforms.Resample(
#             orig_freq=sample_rate,
#             new_freq=TARGET_SAMPLE_RATE
#         )
#         waveform = resampler(waveform)

#     return waveform



import io
import numpy as np
import soundfile as sf
import librosa


def load_audio_from_bytes(audio_bytes: bytes, target_sr: int = 16000):
    """
    Load audio from raw bytes and return waveform + sample rate
    """

    # Read audio from bytes (mp3 / wav)
    with io.BytesIO(audio_bytes) as audio_buffer:
        waveform, sr = sf.read(audio_buffer)

    # Convert stereo → mono
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    # Resample to 16kHz for XLS-R
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return waveform, sr
