from audio_recorder import AudioRecorder

recorder = AudioRecorder()

devices = recorder.get_audio_devices()
print('Verfügbare Eingabegeräte:')
for device in devices:
    print(f"  {device['id']}: {device['name']} ({device['channels']} Kanäle)")