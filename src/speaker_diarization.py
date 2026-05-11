import numpy as np
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
import os

import embedding_encoder


class SpeakerDiarization:
    def __init__(self, num_speakers=None):
        self.num_speakers = num_speakers
        self.segment_duration = 1.0  # 1 Sekunde pro Segment

    def identify_speakers(self, audio_path: str):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio-Datei nicht gefunden: {audio_path}")

        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(np.float32)
        duration = len(audio) / sr
        print(f"[Diarization] Analysiere '{os.path.basename(audio_path)}' ({duration:.1f}s)")

        # Audio auf 16 kHz resamplen (ECAPA-TDNN Anforderung)
        audio_16k = embedding_encoder.resample_to_16k(audio, sr)
        seg_len_16k = 16000  # 1 Sekunde bei 16 kHz
        num_segments = len(audio_16k) // seg_len_16k

        if num_segments < 2:
            print("[Diarization] Audio zu kurz (< 2 Segmente) – ein Sprecher angenommen.")
            return [0] * max(num_segments, 1), {0: np.zeros(192)}


        print(f"[Diarization] Berechne Embeddings für {num_segments} Segmente...")
        embeddings = []
        for i in range(num_segments):
            seg = audio_16k[i * seg_len_16k : (i + 1) * seg_len_16k]
            emb = embedding_encoder.get_embedding(seg, 16000)
            embeddings.append(emb)

        embeddings = np.array(embeddings)  


        original_num_speakers = self.num_speakers
        if self.num_speakers is None:
            if duration < 10:
                n = 2
            elif duration < 45:
                n = min(3, max(2, num_segments // 15))
            else:
                n = min(5, max(2, num_segments // 15))
            self.num_speakers = n
            print(f"[Diarization] Automatische Sprecher-Anzahl: {n} (Dauer: {duration:.0f}s)")

        n_clusters = min(self.num_speakers, num_segments)


        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
        )
        raw_labels = clustering.fit_predict(embeddings)

        unique_labels = sorted(set(raw_labels))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        normalized_labels = [label_map[l] for l in raw_labels]

        cluster_embeddings = {}
        for label in set(normalized_labels):
            cluster_embs = embeddings[[i for i, l in enumerate(normalized_labels) if l == label]]
            mean_emb = np.mean(cluster_embs, axis=0)
            # Erneut normieren
            norm = np.linalg.norm(mean_emb)
            cluster_embeddings[label] = mean_emb / norm if norm > 0 else mean_emb

        print(
            f"[Diarization] Abgeschlossen: {len(set(normalized_labels))} Sprecher-Cluster erkannt. "
            f"Labels: {dict(zip(*np.unique(normalized_labels, return_counts=True)))}"
        )

        # num_speakers zurücksetzen, damit die nächste Aufnahme neu berechnet wird
        self.num_speakers = original_num_speakers

        return normalized_labels, cluster_embeddings