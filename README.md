# Meeting Recorder App

A privacy-first, locally-hosted meeting transcription and speaker recognition tool. Perfect for medical consultations, interviews, and professional meetings.

## Features

- **Audio Recording**: Capture meetings with automatic audio source detection
- **Speech Recognition**: Real-time transcription using OpenAI's Whisper model
- **Speaker Diarization**: Identify and label different speakers in conversations
- **Voice Profiles**: Train the system to recognize your voice and assign speaker identities
- **Chat-like Interface**: Clean, modern UI displaying transcripts as conversation bubbles
- **Local & Private**: All processing happens locally - no cloud uploads, complete data privacy
- **Database Organization**: Organize recordings by folders and dates with SQLite
- **Dashboard**: Manage and search your recorded meetings

## Tech Stack

### Backend
- **Python** - Core application logic
- **Flask** - REST API framework
- **SQLite** - Local database
- **faster-whisper** - Speech recognition (ASR)
- **speechbrain** - Speaker embedding and diarization
- **scikit-learn** - Clustering algorithms for speaker identification
- **sounddevice/soundfile** - Audio recording and processing
- **PyTorch/torchaudio** - Deep learning models

### Frontend
- **HTML/CSS/JavaScript** - Modern responsive UI
- **Dark theme** - Eye-friendly interface
- **Real-time status updates** - Live transcription progress

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd meeting-recorder-app
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the Application

1. **Run the Flask server**
   ```bash
   cd src
   python app.py
   ```

2. **Open in browser**
   Navigate to `http://localhost:5000` - the web interface will load automatically

### Recording a Meeting

1. Navigate to the home page
2. Click "Start Recording"
3. Allow microphone access when prompted
4. Conduct your meeting - audio will be captured automatically
5. Click "Stop Recording" when finished
6. The app will automatically transcribe and identify speakers

### Managing Recordings

- **View Recordings**: Browse all recorded meetings in the dashboard
- **Organize**: Create folders to organize recordings by client, date, or topic
- **Search**: Find recordings using the search functionality
- **Delete**: Remove recordings you no longer need

### Train Voice Profiles

1. Go to voice profiles section
2. Record a sample of your voice
3. The system learns to identify you in future recordings
4. Unknown speakers will be labeled as "Person X"

## Project Structure

```
meeting-recorder-app/
├── src/
│   ├── app.py                  # Main Flask application
│   ├── audio_recorder.py       # Audio recording functionality
│   ├── transcriber.py          # Whisper transcription
│   ├── speaker_diarization.py  # Speaker identification
│   ├── voice_profile.py        # Voice profile management
│   ├── database.py             # Database operations
│   └── embedding_encoder.py    # Audio embedding processing
├── ui/
│   └── index.html              # Web interface
├── data/
│   ├── recordings/             # Stored audio files
│   └── voice_profiles.pkl      # Trained voice profiles
├── requirements.txt            # Python dependencies
├── PROJECT_LOG.md              # Development log
└── README.md                   # This file
```

## API Endpoints

### Recordings
- `GET /api/recordings` - List all recordings
- `POST /api/recordings` - Create new recording
- `PUT /api/recordings/<id>` - Update recording
- `DELETE /api/recordings/<id>` - Delete recording
- `GET /api/recordings/<id>/transcript` - Get transcript

### Audio Control
- `POST /api/recording/start` - Start recording
- `POST /api/recording/stop` - Stop recording
- `GET /api/recording/status` - Get recording status
- `GET /api/audio/devices` - List available audio devices

### Folders
- `GET /api/folders` - List all folders
- `POST /api/folders` - Create folder
- `DELETE /api/folders/<id>` - Delete folder

## Configuration

Audio devices are automatically detected. The app attempts to use Stereo Mix or Virtual Cable if available for better recording quality.

## Performance Notes

- **CPU Mode**: By default, Whisper runs on CPU (slower but works on any machine)
- **GPU Mode**: For faster transcription, ensure CUDA-compatible GPU and PyTorch with CUDA support
- **Model Size**: Using 'small' model for balance between accuracy and speed. Options: tiny, base, small, medium, large

## Troubleshooting

### No audio devices detected
- Check system audio settings
- Ensure microphone is enabled and not muted
- Try selecting a different audio device from available options

### Transcription errors
- Ensure audio quality is good (minimize background noise)
- Check that Whisper model is downloaded (automatic on first run)
- Verify sufficient disk space for model files (~1.5GB for small model)

### Speaker recognition issues
- Train voice profiles with clear, clean audio samples
- Ensure sufficient audio duration for accurate speaker identification
- Multiple speakers may require distinct voice characteristics

## Privacy & Security

- **Local Processing**: All audio files and transcriptions stay on your machine
- **No Cloud Uploads**: No data is sent to external services
- **No Internet Required**: Works completely offline
- **SQLite Database**: Local database for organizing recordings

## Performance Optimization

- Use a modern multi-core processor for faster transcription
- Consider GPU acceleration for large-scale processing
- Close unnecessary applications to free up system resources

## Future Enhancements

- Export transcripts to multiple formats (PDF, Word, SRT)
- Multi-language support
- Real-time transcription during recording
- Advanced search with transcript indexing
- Meeting summaries and insights
- Integration with calendar systems




