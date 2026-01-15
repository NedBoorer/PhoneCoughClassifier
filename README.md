# Voice Health Agent & Phone Cough Classifier

🎙️ **An AI-powered voice agent for accessible respiratory and mental health screening.**

This platform allows users to call a phone number, interact with an AI voice agent in their native language, and receive preliminary health screenings for **Respiratory Diseases (Cough)**, **Parkinson's Disease**, and **Depression** using advanced vocal biomarker analysis.

## ✨ Key Features (Implemented)

### 📞 Voice Interface
- **Twilio Integration**: Full duplex voice conversation handling.
- **Multi-Language Support**: Infrastructure for 10 distinct Indian languages (English, Hindi, Tamil, Telugu, etc.) for rural accessibility.
- **Intelligent Routing**: Menu system to direct users to specific screening modules.

### 🏥 Disease Screening Modules

#### 1. Respiratory & Cough Analysis
- **Model**: Google HeAR (Health Acoustic Representations) embeddings + Scikit-learn classifier.
- ** capabilities**: Distinguishes between Dry, Wet, Whooping, and Chronic coughs.
- **Dataset**: Trained on the COUGHVID dataset (30K+ samples).

#### 2. Parkinson's Disease Detection
- **Biomarkers**: Jitter, Shimmer, HNR (Harmonics-to-Noise Ratio), F0 variability.
- **Protocol**: Analyzes sustained vowel phonation (users saying "Aaaah").
- **Analysis**: Rule-based clinical thresholds integrated with Scikit-learn classifier support.

#### 3. Depression Screening
- **Biomarkers**: Speech rate, pause duration, pitch variability, spectral centroid.
- **Protocol**: Analyzes spontaneous speech patterns.
- **Analysis**: Detects "flat affect" and psychomotor retardation markers.

### 🏗️ Core Infrastructure
- **FastAPI Backend**: High-performance async Python web framework.
- **Database**: SQLAlchemy + SQLite (async) for tracking Call Records, Patient Info, and Assessments.
- **Audio Processing**: 
    - `librosa` and `scipy` for signal processing.
    - `opensmile` integration for professional-grade feature extraction.
- **Model Hub**: Centralized manager for loading and running multiple diagnostic models.

---

## 🚀 Work In Progress & Roadmap

### ⚠️ Immediate To-Dos
- [ ] **Model Training**: 
    - Fine-tune Parkinson's classifier on real datasets (e.g., MDVR-KCL).
    - Train Depression classifier on DAIC-WOZ or similar datasets.
    - *Current status*: Using rule-based fallbacks and clinical thresholds.
- [ ] **India Webhooks**: Complete the implementation of `india_webhooks.py` to fully enable the localized IVR flows.
- [ ] **Unit Testing**: Increase test coverage for the ML pipeline and webhook logic.

### 🔮 Future Improvements
- **Clinical Validation**: Rigorous testing against medically verified ground truth data.
- **Deployment**: Dockerize the application and deploy to a cloud provider (AWS/GCP).
- **Frontend Dashboard**: Create a web UI for doctors/admins to review cases and listen to recordings.
- **WhatsApp Integration**: Send detailed reports via WhatsApp API.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- `ffmpeg` (installed via brew/apt for audio processing)
- Twilio Account

### 1. Clone and Install
```bash
git clone https://github.com/YourUsername/PhoneCoughClassifier.git
cd PhoneCoughClassifier
pip install -r requirements.txt
python setup.py
```

### 2. Environment Configuration
Copy the example env file and fill in your credentials:
```bash
cp .env.example .env
```
Update `.env` with:
- `TWILIO_ACCOUNT_SID` & `TWILIO_AUTH_TOKEN`
- `OPENAI_API_KEY` (for conversation generation)

### 3. Run the Server
```bash
python -m uvicorn app.main:app --reload
```

### 4. Expose to Internet (for Twilio)
Use ngrok to expose your local server:
```bash
ngrok http 8000
```
Update your Twilio Voice Webhook to: `https://<your-ngrok-url>/twilio/voice/incoming`

---

## 🏗️ Project Architecture

```
PhoneCoughClassifier/
├── app/
│   ├── api/                 # Webhook endpoints (Twilio, Health, India)
│   ├── database/            # SQL Models (Calls, Patients, Results)
│   ├── ml/                  # Machine Learning Modules
│   │   ├── classifier.py         # Cough Classifier
│   │   ├── parkinsons_classifier.py
│   │   ├── depression_classifier.py
│   │   ├── voice_biomarkers.py   # Feature Extraction (Jitter/Shimmer/etc)
│   │   └── model_hub.py          # Unified Model interface
│   └── services/            # External services (Twilio, S3)
├── external_models/         # Submodules for specific model architectures
├── data/                    # Local datasets (not in git)
├── scripts/                 # Training and setup scripts
└── recordings/              # Audio storage
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/twilio/voice/incoming` | Main entry point for calls |
| `POST` | `/health/parkinsons/incoming` | Parkinson's specific screening flow |
| `POST` | `/health/depression/incoming` | Depression specific screening flow |
| `POST` | `/test/classify` | Test classification with file upload |
| `GET`  | `/health` | System health and model status |

---

## 🔒 Privacy & Safety
- **No Diagnosis**: This tool provides a risk assessment/screening only, NOT a medical diagnosis.
- **Data Retention**: Audio files are processed and optionally deleted based on configuration.
- **HIPAA**: Designed with privacy in mind, hashing phone numbers and isolating PII.

