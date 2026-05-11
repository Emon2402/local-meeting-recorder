import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from database import get_db_connection, init_db
from audio_recorder import AudioRecorder
from transcriber import WhisperTranscriber
from speaker_diarization import SpeakerDiarization
from voice_profile import VoiceProfile
from embedding_encoder import cosine_similarity
import sounddevice as sd
from datetime import datetime
import traceback

ui_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ui')
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

init_db()

# Auto-Erkennung für beste Audio-Quelle (Stereo-Mix, Virtual Cable, etc.)
audio_recorder = AudioRecorder(channels=2, auto_detect=True)

transcriber = WhisperTranscriber(model_size='small', device='cpu')

speaker_diarization = SpeakerDiarization(num_speakers=None)

# Status für laufende Transkription
processing_status = {
    'is_processing': False,
    'recording_id': None,
    'progress': ''
}

voice_profile = VoiceProfile()

AUDIO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'recordings')
os.makedirs(AUDIO_DIR, exist_ok=True)

VOICE_PROFILES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'voice_profiles.pkl')

if os.path.exists(VOICE_PROFILES_PATH):
    voice_profile.load_profiles(VOICE_PROFILES_PATH)

def format_transcript_with_speakers(
    transcript_segments,
    speaker_labels,
    cluster_embeddings=None,
    segment_duration=1.0,
):
    
    if not speaker_labels:
        if isinstance(transcript_segments, list):
            return " ".join([seg['text'] for seg in transcript_segments])
        return str(transcript_segments)

    if not transcript_segments:
        return "[Keine Sprache erkannt oder Audio zu kurz]"

    
    speaker_clusters = {
        label: f"Person {label + 1}" for label in set(speaker_labels)
    }

    print(f"[Transcript] Profile vorhanden: {list(voice_profile.profiles.keys())}")
    print(f"[Transcript] Cluster vor Identifikation: {speaker_clusters}")

    # Identifikation mit ECAPA-TDNN Cosinus-Ähnlichkeit
    if cluster_embeddings and voice_profile.profiles:
        matches = []
        for label, embedding in cluster_embeddings.items():
            for name, profile in voice_profile.profiles.items():
                sim = cosine_similarity(embedding, profile["embedding"])
                matches.append((sim, label, name))
                
        # Nach höchster Ähnlichkeit absteigend sortieren
        matches.sort(reverse=True, key=lambda x: x[0])
        
        assigned_labels = set()
        assigned_names = set()
        
        for sim, label, name in matches:
            if sim >= 0.75:
                if label not in assigned_labels and name not in assigned_names:
                    speaker_clusters[label] = name
                    assigned_labels.add(label)
                    assigned_names.add(name)
                    print(f"[Transcript] Cluster {label} exklusiv zugewiesen an '{name}' (Cosinus: {sim:.3f})")
            else:
                
                pass

    print(f"[Transcript] Cluster nach Identifikation: {speaker_clusters}")


    formatted_text = ""
    current_speaker = None

    for segment in transcript_segments:
        start_idx = int(segment['start'] / segment_duration)
        end_idx   = int(segment['end']   / segment_duration)

        segment_labels = speaker_labels[start_idx : end_idx + 1]
        if segment_labels:
            speaker_id = max(set(segment_labels), key=segment_labels.count)
        else:
            idx = min(start_idx, len(speaker_labels) - 1)
            speaker_id = speaker_labels[idx] if idx < len(speaker_labels) else 0

        speaker_name = speaker_clusters.get(speaker_id, f"Person {speaker_id + 1}")

        if current_speaker != speaker_name:
            if formatted_text:
                formatted_text += "\n"
            formatted_text += f"{speaker_name}: "
            current_speaker = speaker_name
        elif formatted_text and not formatted_text.endswith(" "):
            formatted_text += " "

        formatted_text += segment['text'].strip() + " "

    return formatted_text.strip()


@app.route('/')
def home():
    return send_from_directory(ui_folder, 'index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

# --- API Endpunkte für Audio-Geräte ---

@app.route('/api/audio/devices', methods=['GET'])
def get_audio_devices():
    devices = audio_recorder.get_audio_devices()
    current_device = audio_recorder.device_id
    default_device = sd.default.device[0]
    
    return jsonify({
        'devices': devices,
        'current_device': current_device,
        'default_device': default_device
    })

@app.route('/api/audio/set-device/<int:device_id>', methods=['POST'])
def set_audio_device(device_id):
    try:
        # Stoppe aktuelle Aufnahme falls laufen
        if audio_recorder.is_recording:
            audio_recorder.stop_recording()
        
        # Setze neues Gerät
        audio_recorder.device_id = device_id
        audio_recorder._validate_device_channels()
        
        device_info = sd.query_devices(device_id)
        return jsonify({
            'message': f'Audio-Gerät gewechselt zu: {device_info["name"]}',
            'device_id': device_id,
            'channels': audio_recorder.channels
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/audio/info', methods=['GET'])
def get_audio_info():
    try:
        device_info = sd.query_devices(audio_recorder.device_id)
        current_device_name = device_info['name']
    except:
        current_device_name = "Standard"
    
    return jsonify({
        'device_id': audio_recorder.device_id,
        'device_name': current_device_name,
        'channels': audio_recorder.channels,
        'sample_rate': audio_recorder.sample_rate,
        'is_recording': audio_recorder.is_recording,
        'supports_stereo_mix': any('Stereo Mix' in d['name'] or 'VB-Audio' in d['name'] 
                                   for d in audio_recorder.get_audio_devices()),
        'hint': 'Falls „supports_stereo_mix" false ist: Aktiviere Stereo-Mix in Windows-Soundeinstellungen!'
    })

# --- API Endpunkte für Ordner ---


@app.route('/api/folders', methods=['GET'])
def get_folders():
    
    conn = get_db_connection()
    folders = conn.execute('SELECT * FROM folders ORDER BY created_at DESC').fetchall()
    conn.close()
    return jsonify([dict(folder) for folder in folders])

@app.route('/api/folders', methods=['POST'])
def create_folder():
    
    data = request.get_json()
    name = data.get('name')
    
    if not name:
        return jsonify({'error': 'Name ist erforderlich'}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO folders (name) VALUES (?)', (name,))
    conn.commit()
    folder_id = cursor.lastrowid
    conn.close()
    
    return jsonify({'id': folder_id, 'name': name}), 201

@app.route('/api/folders/<int:folder_id>', methods=['DELETE'])
def delete_folder(folder_id):
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('UPDATE recordings SET folder_id = NULL WHERE folder_id = ?', (folder_id,))
    
    cursor.execute('DELETE FROM folders WHERE id = ?', (folder_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Ordner gelöscht'})

# --- API Endpunkte für Aufzeichnungen ---

@app.route('/api/recordings', methods=['GET'])
def get_recordings():
    
    folder_id = request.args.get('folder_id')
    conn = get_db_connection()
    
    if folder_id:
        recordings = conn.execute(
            'SELECT * FROM recordings WHERE folder_id = ? ORDER BY date DESC', 
            (folder_id,)
        ).fetchall()
    else:
        recordings = conn.execute(
            'SELECT * FROM recordings ORDER BY date DESC'
        ).fetchall()
    
    conn.close()
    return jsonify([dict(rec) for rec in recordings])

@app.route('/api/recordings', methods=['POST'])
def create_recording():
    
    data = request.get_json()
    title = data.get('title')
    folder_id = data.get('folder_id')
    
    if not title:
        return jsonify({'error': 'Titel ist erforderlich'}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO recordings (title, folder_id) VALUES (?, ?)',
        (title, folder_id)
    )
    conn.commit()
    recording_id = cursor.lastrowid
    conn.close()
    
    return jsonify({'id': recording_id, 'title': title, 'folder_id': folder_id}), 201

@app.route('/api/recordings/<int:recording_id>', methods=['PUT'])
def update_recording(recording_id):
    
    data = request.get_json()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if 'title' in data:
        cursor.execute('UPDATE recordings SET title = ? WHERE id = ?', (data['title'], recording_id))
        
    if 'folder_id' in data:
        cursor.execute('UPDATE recordings SET folder_id = ? WHERE id = ?', (data['folder_id'], recording_id))
        
    conn.commit()
    conn.close()
    return jsonify({'message': 'Aufzeichnung aktualisiert'})

@app.route('/api/recordings/<int:recording_id>', methods=['DELETE'])
def delete_recording(recording_id):
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM recordings WHERE id = ?', (recording_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Aufzeichnung gelöscht'})

# --- API Endpunkte für Audio-Aufzeichnung ---

@app.route('/api/recording/start', methods=['POST'])
def start_recording():
   
    try:
        data = request.get_json()
        title = data.get('title', 'Neue Aufzeichnung')
        folder_id = data.get('folder_id')
        delete_audio = data.get('delete_audio', False)
        
        print(f"Starte Aufzeichnung: title={title}, folder_id={folder_id}, delete_audio={delete_audio}")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO recordings (title, folder_id, delete_audio) VALUES (?, ?, ?)',
            (title, folder_id, delete_audio)
        )
        conn.commit()
        recording_id = cursor.lastrowid
        conn.close()
        
        print(f"Aufzeichnung in Datenbank erstellt: ID={recording_id}")
        
        success = audio_recorder.start_recording()
        
        print(f"Audio-Aufnahme gestartet: success={success}")
        
        if success:
            return jsonify({
                'message': 'Aufzeichnung gestartet',
                'recording_id': recording_id,
                'title': title
            }), 200
        else:
            return jsonify({'error': 'Aufnahme läuft bereits'}), 400
    except Exception as e:
        print(f"Fehler beim Starten der Aufzeichnung: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/recording/stop', methods=['POST'])
def stop_recording():
    
    data = request.get_json()
    recording_id = data.get('recording_id')
    
    if not recording_id:
        return jsonify({'error': 'recording_id ist erforderlich'}), 400
    
    audio_data = audio_recorder.stop_recording()
    
    if audio_data is None:
        return jsonify({'error': 'Keine Aufnahme läuft'}), 400
    
    conn = get_db_connection()
    recording = conn.execute('SELECT delete_audio FROM recordings WHERE id = ?', (recording_id,)).fetchone()
    delete_audio = recording['delete_audio'] if recording else False
    
    filepath = None
    
    if not delete_audio:
        filename = f"recording_{recording_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = os.path.join(AUDIO_DIR, filename)
        
        if audio_recorder.save_audio(audio_data, filepath):
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE recordings SET audio_path = ? WHERE id = ?',
                (filepath, recording_id)
            )
            conn.commit()
        else:
            conn.close()
            return jsonify({'error': 'Fehler beim Speichern der Audio-Datei'}), 500
    else:
        print(f"Audio für Aufzeichnung {recording_id} wird nicht gespeichert (delete_audio=True)")
        
        temp_filename = f"temp_recording_{recording_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        temp_filepath = os.path.join(AUDIO_DIR, temp_filename)
        
        if audio_recorder.save_audio(audio_data, temp_filepath):
            filepath = temp_filepath
        else:
            conn.close()
            return jsonify({'error': 'Fehler beim Speichern der temporären Audio-Datei'}), 500
    
    # Transkription und Sprechererkennung durchführen
    try:
        # Setze Processing-Status
        processing_status['is_processing'] = True
        processing_status['recording_id'] = recording_id
        processing_status['progress'] = 'Transkribiere Audio...'
        
        print(f"Starte Transkription für Aufzeichnung {recording_id}...")
        transcript_segments = transcriber.transcribe(filepath, language='de')
        
        processing_status['progress'] = 'Erkenne Sprecher...'
        print(f"Starte Sprechererkennung für Aufzeichnung {recording_id}...")
        speaker_labels, cluster_embeddings = speaker_diarization.identify_speakers(filepath)
        
        try:
            formatted_transcript = format_transcript_with_speakers(
                transcript_segments,
                speaker_labels,
                cluster_embeddings=cluster_embeddings,
                segment_duration=speaker_diarization.segment_duration,
            )
        except Exception as fmt_e:
            print(f"Fehler bei der Sprecher-Formatierung: {fmt_e}")
            traceback.print_exc()
            if transcript_segments:
                formatted_transcript = " ".join([seg['text'] for seg in transcript_segments])
            else:
                formatted_transcript = "[Keine Sprache erkannt oder Audio zu kurz]"
        
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE recordings SET transcript_path = ? WHERE id = ?',
            (formatted_transcript, recording_id)
        )
        conn.commit()
        
        print(f"Transkription und Sprechererkennung für Aufzeichnung {recording_id} abgeschlossen")
        
        if delete_audio and os.path.exists(filepath):
            os.remove(filepath)
            print(f"Temporäre Audio-Datei gelöscht: {filepath}")
            filepath = None
            
    except Exception as e:
        print(f"Fehler bei der Transkription: {e}")
        traceback.print_exc()
    finally:
        # Setze Processing-Status zurück
        processing_status['is_processing'] = False
        processing_status['recording_id'] = None
        processing_status['progress'] = ''
    
    conn.close()
    
    return jsonify({
        'message': 'Aufzeichnung gestoppt' + (' und gespeichert' if not delete_audio else ' (Audio nicht gespeichert)'),
        'recording_id': recording_id,
        'audio_path': filepath
    }), 200

@app.route('/api/recording/status', methods=['GET'])
def recording_status():
    
    return jsonify({
        'is_recording': audio_recorder.is_recording,
        'is_processing': processing_status['is_processing'],
        'recording_id': processing_status['recording_id'],
        'progress': processing_status['progress']
    })

@app.route('/api/recordings/<int:recording_id>/transcript', methods=['GET'])
def get_transcript(recording_id):
    
    conn = get_db_connection()
    recording = conn.execute('SELECT transcript_path FROM recordings WHERE id = ?', (recording_id,)).fetchone()
    conn.close()
    
    if recording:
        transcript = recording['transcript_path'] or ''
        return jsonify({
            'recording_id': recording_id,
            'transcript': transcript
        })
    else:
        return jsonify({'error': 'Aufzeichnung nicht gefunden'}), 404

# --- API Endpunkte für Stimmen-Training ---

@app.route('/api/voice-profiles', methods=['GET'])
def get_voice_profiles():
    
    conn = get_db_connection()
    users = conn.execute('SELECT id, name, created_at FROM users ORDER BY created_at DESC').fetchall()
    conn.close()
    return jsonify([dict(user) for user in users])

@app.route('/api/voice-profiles', methods=['POST'])
def create_voice_profile():
   
    data = request.get_json()
    name = data.get('name')
    audio_path = data.get('audio_path')
    
    if not name:
        return jsonify({'error': 'Name ist erforderlich'}), 400
    
    if not audio_path:
        return jsonify({'error': 'Audio-Pfad ist erforderlich'}), 400
    
    try:
        features = voice_profile.train_profile(audio_path, name)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO users (name, training_audio_path, features) VALUES (?, ?, ?)',
            (name, audio_path, str(features.tolist()))
        )
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        voice_profile.save_profiles(VOICE_PROFILES_PATH)
        
        return jsonify({
            'message': 'Stimmen-Profil erstellt',
            'user_id': user_id,
            'name': name
        }), 201
    except Exception as e:
        print(f"Fehler beim Erstellen des Stimmen-Profils: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice-profiles/<int:user_id>', methods=['DELETE'])
def delete_voice_profile(user_id):
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    user = conn.execute('SELECT name FROM users WHERE id = ?', (user_id,)).fetchone()
    
    if not user:
        conn.close()
        return jsonify({'error': 'Benutzer nicht gefunden'}), 404
    
    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    
    if user['name'] in voice_profile.profiles:
        del voice_profile.profiles[user['name']]
        voice_profile.save_profiles(VOICE_PROFILES_PATH)
    
    return jsonify({'message': 'Stimmen-Profil gelöscht'})

@app.route('/api/voice-profiles/identify', methods=['POST'])
def identify_speaker():
    
    data = request.get_json()
    audio_path = data.get('audio_path')
    
    if not audio_path:
        return jsonify({'error': 'Audio-Pfad ist erforderlich'}), 400
    
    try:
        speaker = voice_profile.identify_speaker(audio_path)
        
        if speaker:
            return jsonify({
                'speaker': speaker,
                'identified': True
            })
        else:
            return jsonify({
                'speaker': None,
                'identified': False,
                'message': 'Kein bekannter Sprecher erkannt'
            })
    except Exception as e:
        print(f"Fehler bei der Sprecher-Identifikation: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/<path:path>', methods=['GET'])
def serve_static_or_frontend(path):
    if path.startswith('api/'):
        return jsonify({'error': 'API endpoint not found'}), 404
    file_path = os.path.join(ui_folder, path)
    if os.path.isfile(file_path):
        return send_from_directory(ui_folder, path)
    return send_from_directory(ui_folder, 'index.html')

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)