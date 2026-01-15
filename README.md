# Phone Cough Classifier

🎙️ **Voice agent pipeline for cough classification using real datasets**

Users call a phone number, have a brief conversation, cough into the phone, and receive their classification result via SMS.

## ✨ Features

- **Voice Agent**: Full phone call flow with Twilio integration
- **Real ML Model**: Trained on COUGHVID dataset (30K+ samples) or Google HeAR embeddings
- **Multi-Language**: 10 Indian languages for rural accessibility
- **Audio Quality**: SNR estimation, clipping detection, quality recommendations
- **SMS Results**: Automated result delivery with health recommendations

## 🚀 Quick Start

```bash
# 1. Clone and setup
cd PhoneCoughClassifier
python setup.py

# 2. Edit .env with your credentials
# (Twilio, OpenAI API keys)

# 3. Start server
python -m uvicorn app.main:app --reload --port 8000

# 4. Test the API
open http://localhost:8000/docs
```

## 🏗️ Architecture

```
User → Phone Call → Twilio → FastAPI Webhooks → Record Cough
                                      ↓
      SMS Result ← Twilio ← Classification ← Audio Processing
```

## 📁 Project Structure

```
PhoneCoughClassifier/
├── app/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Settings management
│   ├── api/
│   │   ├── twilio_webhooks.py  # Voice call handlers
│   │   ├── india_webhooks.py   # Multi-language IVR
│   │   └── test_endpoints.py   # Testing without calls
│   ├── ml/
│   │   ├── classifier.py       # Main cough classifier
│   │   └── feature_extractor.py # Audio feature extraction
│   ├── database/
│   │   ├── database.py         # SQLAlchemy async
│   │   └── models.py           # Data models
│   ├── services/
│   │   └── twilio_service.py   # SMS, recording download
│   └── utils/
│       ├── audio_processing.py # Format conversion
│       ├── audio_quality.py    # Quality assessment
│       └── i18n.py             # 10 language translations
├── scripts/
│   ├── download_coughvid.py    # Download training data
│   └── train_model.py          # Train classifier
├── data/                       # Datasets (gitignored)
├── models/                     # Trained models
├── recordings/                 # Call recordings
├── setup.py                    # One-command setup
├── requirements.txt            # Python dependencies
└── .env.example                # Configuration template
```

## 🔧 Configuration

Copy `.env.example` to `.env` and fill in:

```bash
# Twilio
TWILIO_ACCOUNT_SID=ACxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890

# OpenAI (for conversation)
OPENAI_API_KEY=sk-xxxxxxxx

# Webhook URL (ngrok for local dev)
BASE_URL=https://your-domain.ngrok.io
```

## 📱 Twilio Setup

1. Get a Twilio phone number
2. Set Voice webhook to: `https://your-domain/twilio/voice/incoming`
3. For India multi-language: `https://your-domain/india/voice/incoming`

## 🧪 Testing

### Without Phone Calls

Upload audio files directly via `/test/classify`:

```bash
curl -X POST http://localhost:8000/test/classify \
  -F "audio_file=@cough.wav"
```

### API Documentation

Open http://localhost:8000/docs for Swagger UI

## 📊 ML Models

### Option 1: COUGHVID (Recommended)

```bash
# Download dataset (requires Kaggle API)
python scripts/download_coughvid.py --output data/coughvid

# Train classifier
python scripts/train_model.py --data-dir data/coughvid
```

### Option 2: Synthetic Data (Demo)

```bash
python scripts/train_model.py --use-synthetic
```

## 🌍 Supported Languages

| Code | Language   | Native     |
|------|------------|------------|
| en   | English    | English    |
| hi   | Hindi      | हिंदी      |
| ta   | Tamil      | தமிழ்      |
| te   | Telugu     | తెలుగు     |
| bn   | Bengali    | বাংলা      |
| mr   | Marathi    | मराठी      |
| gu   | Gujarati   | ગુજરાતી    |
| kn   | Kannada    | ಕನ್ನಡ      |
| ml   | Malayalam  | മലയാളം     |
| pa   | Punjabi    | ਪੰਜਾਬੀ     |

## 📋 Classification Types

| Type      | Description                          |
|-----------|--------------------------------------|
| Dry       | Non-productive, tickly sensation     |
| Wet       | Productive, contains mucus/phlegm    |
| Whooping  | Barking sound, possible pertussis    |
| Chronic   | Persistent cough (>3 weeks)          |
| Normal    | Typical acute cough, likely viral    |

## 🔒 Data Privacy

- Audio deleted after classification (configurable)
- Caller numbers hashed in database
- HIPAA-compliant recommendations only
- No medical diagnosis provided

## 📈 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/test/classify` | POST | Upload and classify audio |
| `/test/quality` | POST | Check audio quality |
| `/test/status` | GET | System component status |
| `/twilio/voice/incoming` | POST | Handle incoming calls |
| `/india/voice/incoming` | POST | Multi-language call handler |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License - See LICENSE file

---

Built with ❤️ using FastAPI, Twilio, and COUGHVID dataset
