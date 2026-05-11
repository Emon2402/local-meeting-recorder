import numpy as np
import soundfile as sf
import os
import pickle

import embedding_encoder


class VoiceProfile:
    _FORMAT_VERSION = "ecapa-tdnn-v1"

    def __init__(self):
        self.profiles = {}


    def train_profile(self, audio_path: str, user_name: str) -> np.ndarray:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio-Datei nicht gefunden: {audio_path}")

        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        print(f"[VoiceProfile] Trainiere Profil für '{user_name}' ({len(audio)/sr:.1f}s Audio)...")

        embedding = embedding_encoder.get_embedding(audio.astype(np.float32), sr)

        self.profiles[user_name] = {
            "embedding": embedding,
            "version": self._FORMAT_VERSION,
        }

        print(f"[VoiceProfile] Profil für '{user_name}' gespeichert (Dim: {embedding.shape}).")
        return embedding


    def identify_speaker_from_embedding(
        self,
        embedding: np.ndarray,
        threshold: float = 0.75,
    ):
        if not self.profiles:
            return None, 0.0

        best_name = None
        best_sim = -1.0

        for name, profile in self.profiles.items():
            sim = embedding_encoder.cosine_similarity(embedding, profile["embedding"])
            print(f"[VoiceProfile] Cosinus-Ähnlichkeit zu '{name}': {sim:.3f}")
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= threshold:
            print(f"[VoiceProfile] Erkannt als '{best_name}' (Ähnlichkeit: {best_sim:.3f})")
            return best_name, best_sim
        else:
            print(f"[VoiceProfile] Nicht erkannt – beste Ähnlichkeit {best_sim:.3f} < {threshold}")
            return None, best_sim

    def identify_speaker(self, audio_path: str, threshold: float = 0.75):
        if not self.profiles:
            return None, 0.0

        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        embedding = embedding_encoder.get_embedding(audio.astype(np.float32), sr)
        return self.identify_speaker_from_embedding(embedding, threshold)


    def save_profiles(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self.profiles, f)
        print(f"[VoiceProfile] Profile gespeichert: {filepath}")

    def load_profiles(self, filepath: str):
        if not os.path.exists(filepath):
            print(f"[VoiceProfile] Keine Profile gefunden: {filepath}")
            return

        with open(filepath, "rb") as f:
            loaded = pickle.load(f)

        valid = {}
        for name, profile in loaded.items():
            if isinstance(profile, dict) and "embedding" in profile:
                valid[name] = profile
                print(f"[VoiceProfile] Profil '{name}' geladen (ECAPA-TDNN Format).")
            else:
                print(
                    f"[VoiceProfile] ⚠ Altes Profil für '{name}' ignoriert – "
                    "bitte Stimmprofil neu aufnehmen!"
                )

        self.profiles = valid

        if not valid:
            print("[VoiceProfile] Keine gültigen Profile vorhanden.")