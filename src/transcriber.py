from faster_whisper import WhisperModel
import os

class WhisperTranscriber:
    def __init__(self, model_size='base', device='cpu'):
        self.model_size = model_size
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        print(f"Lade Whisper-Modell ({self.model_size})...")
        # Modelle werden automatisch heruntergeladen, falls nicht vorhanden
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type="int8"  
        )
        print(f"Whisper-Modell ({self.model_size}) geladen!")
    
    def transcribe(self, audio_path, language='de'):
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio-Datei nicht gefunden: {audio_path}")
        
        print(f"Transkribiere: {audio_path}")
        
        # Audio transkribieren
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True  
        )
        
        print(f"Erkannte Sprache: {info.language} (Wahrscheinlichkeit: {info.language_probability:.2f})")
        print(f"Dauer: {info.duration:.2f} Sekunden")
        
        # Segmente mit Zeitstempeln speichern
        transcript_segments = []
        for segment in segments:
            transcript_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
        
        print(f"Transkription abgeschlossen. {len(transcript_segments)} Segmente gefunden.")
        
        return transcript_segments

if __name__ == '__main__':
    # Testen des Transkribers
    transcriber = WhisperTranscriber(model_size='base')
    
    # Beispiel-Transkription 
    test_audio = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'recordings', 'test.wav')
    if os.path.exists(test_audio):
        segments = transcriber.transcribe(test_audio)
        print("\nTranskript Segmente:")
        for seg in segments:
            print(f"[{seg['start']:.2f}s -> {seg['end']:.2f}s] {seg['text']}")
    else:
        print(f"Keine Test-Datei gefunden: {test_audio}")