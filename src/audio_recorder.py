import sounddevice as sd
import numpy as np
import soundfile as sf
import os
from datetime import datetime
from threading import Thread
import queue

class AudioRecorder:
    def __init__(self, sample_rate=44100, channels=None, device=None, auto_detect=True):
        self.sample_rate = sample_rate
        self.channels = channels if channels is not None else 2  # Standard: Stereo
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread = None
        self.audio_data = []
        self.device_id = device
        self.auto_detect = auto_detect
        
        # Auto-Geräteerkennung
        if self.auto_detect and device is None:
            self.device_id = self.auto_select_audio_device()
        
        # Verfügbare Kanäle prüfen
        if self.device_id is not None:
            self._validate_device_channels()
    
    def _validate_device_channels(self):
    
        try:
            device_info = sd.query_devices(self.device_id)
            max_channels = device_info['max_input_channels']
            
            if self.channels > max_channels:
                print(f"[WARNING] Device '{device_info['name']}' supports only {max_channels} channels.")
                print(f"   Reducing to {max_channels} channel(s)")
                self.channels = max_channels
        except Exception as e:
            print(f"Fehler bei der Geräteprüfung: {e}")
    
    def auto_select_audio_device(self):
       
        devices = sd.query_devices()
        priority_keywords = [
            'Stereo Mix',      # Windows - System Audio
            'Stereo Mix (loopback)',
            'VB-Audio',        # Virtual Audio Cable
            'VirtualAudio',
            'CABLE Output',    # VB-Cable Alternative
            'Loopback',        # Mac
            'What U Hear',     # Ältere Windows
            'Microphone Array',  # Teams-optimiert
        ]
        
        # Erste Priorität: Stereo-Mix oder Virtual Cable
        for keywords in priority_keywords:
            for i, device in enumerate(devices):
                if device['max_input_channels'] >= 2 and keywords.lower() in device['name'].lower():
                    print(f"[OK] Audio source found: {device['name']} (ID: {i}, {device['max_input_channels']} channels)")
                    return i
        
        # Fallback: Hochste Stereo-Quelle
        default_device_id = sd.default.device[0]
        best_device_id = default_device_id
        max_channels = 0
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > max_channels:
                max_channels = device['max_input_channels']
                best_device_id = i
        
        if best_device_id is not None and max_channels >= 2:
            device_info = sd.query_devices(best_device_id)
            print(f"[INFO] Stereo-Mix not found. Using: {device_info['name']} ({max_channels} channels)")
            print(f"    Hint: Enable 'Stereo-Mix' in Windows sound settings for better results!")
            return best_device_id
        
        # Fallback auf Standard-Mikrofon (Mono)
        print(f"[WARNING] Only Mono audio available. Teams recordings may not work optimally!")
        return default_device_id
    
    def _audio_callback(self, indata, frames, time, status):
       
        if status:
            print(f"Audio-Callback Status: {status}")
        self.audio_queue.put(indata.copy())
    
    def start_recording(self):
        if self.is_recording:
            print("Aufnahme läuft bereits")
            return False
        
        self.is_recording = True
        self.audio_data = []
        self.audio_queue = queue.Queue()
        
        try:
            device_info = sd.query_devices(self.device_id) if self.device_id is not None else "Default"
            print(f"[RECORDING] Using audio source: {device_info['name'] if isinstance(device_info, dict) else device_info}")
            print(f"   Channels: {self.channels}, Sample-Rate: {self.sample_rate} Hz")
        except:
            pass
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.device_id,
            callback=self._audio_callback
        )
        self.stream.start()
        
        self.recording_thread = Thread(target=self._collect_audio)
        self.recording_thread.start()
        
        print("[OK] Recording started")
        return True
    
    def _collect_audio(self):
        while self.is_recording:
            try:
                data = self.audio_queue.get(timeout=0.1)
                self.audio_data.append(data)
            except queue.Empty:
                continue
    
    def stop_recording(self):
        if not self.is_recording:
            print("Keine Aufnahme läuft")
            return None
        
        self.is_recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        if self.recording_thread:
            self.recording_thread.join(timeout=2)
        
        if self.audio_data:
            audio_array = np.concatenate(self.audio_data, axis=0)
            print(f"Aufnahme gestoppt. Dauer: {len(audio_array) / self.sample_rate:.2f} Sekunden")
            return audio_array
        else:
            print("Keine Audio-Daten aufgenommen")
            return None
    
    def save_audio(self, audio_data, filename):
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            sf.write(filename, audio_data, self.sample_rate)
            print(f"Audio gespeichert: {filename}")
            return True
        except Exception as e:
            print(f"Fehler beim Speichern des Audio: {e}")
            return False
    
    def get_audio_devices(self):
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels']
                })
        
        return input_devices

if __name__ == '__main__':
    recorder = AudioRecorder()
    
    print("Verfügbare Eingabegeräte:")
    devices = recorder.get_audio_devices()
    for device in devices:
        print(f"  {device['id']}: {device['name']} ({device['channels']} Kanäle)")
    
    print("\nDrücke ENTER um die Aufnahme zu starten...")
    input()
    
    print("Aufnahme läuft... Drücke ENTER um zu stoppen.")
    recorder.start_recording()
    input()
    
    audio_data = recorder.stop_recording()
    
    if audio_data is not None:
        filename = f"test_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'recordings', filename)
        recorder.save_audio(audio_data, filepath)