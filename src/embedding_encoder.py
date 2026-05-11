import os
import numpy as np
import torch
from scipy import signal as scipy_signal

# Pfad für das lokal gecachte Modell 
_MODEL_SAVEDIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "pretrained_models",
    "spkrec-ecapa-voxceleb"
)

_encoder = None  # Globale Singleton-Instanz


def _load_encoder():
    global _encoder
    if _encoder is not None:
        return _encoder

    print("[Encoder] Lade ECAPA-TDNN Speaker Encoder (einmalig)...")

    try:
        from speechbrain.inference.speaker import SpeakerRecognition
    except ImportError:
        from speechbrain.pretrained import SpeakerRecognition

    from speechbrain.utils.fetching import LocalStrategy

    # HuggingFace Snapshot-Cache direkt als savedir verwenden
    hf_cache_base = os.path.join(
        os.environ.get("USERPROFILE", os.path.expanduser("~")),
        ".cache", "huggingface", "hub",
        "models--speechbrain--spkrec-ecapa-voxceleb", "snapshots"
    )
    savedir = _MODEL_SAVEDIR
    if os.path.isdir(hf_cache_base):
        snaps = sorted(os.listdir(hf_cache_base))
        if snaps:
            savedir = os.path.join(hf_cache_base, snaps[-1])
            print(f"[Encoder] Nutze gecachten HF-Snapshot: {savedir}")
    os.makedirs(savedir, exist_ok=True)

    _encoder = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=savedir,
        run_opts={"device": "cpu"},
        local_strategy=LocalStrategy.COPY,
    )
    print("[Encoder] ECAPA-TDNN bereit!")
    return _encoder


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    target_sr = 16000
    if orig_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    return scipy_signal.resample(audio, num_samples)


def get_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    encoder = _load_encoder()

    # Auf 16 kHz resamplen
    audio_16k = resample_to_16k(audio, sample_rate)

    # Mindestlänge: 0.1 Sekunden bei 16 kHz = 1600 Samples
    if len(audio_16k) < 1600:
        # Padding mit Nullen
        audio_16k = np.pad(audio_16k, (0, 1600 - len(audio_16k)))

    audio_tensor = torch.FloatTensor(audio_16k).unsqueeze(0)

    with torch.no_grad():
        embedding = encoder.encode_batch(audio_tensor)

    emb = embedding.squeeze().numpy()

    # L2-Normierung für stabile Cosinus-Ähnlichkeit
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.dot(emb1, emb2))
