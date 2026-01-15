# Voice Health Agent & Phone Cough Classifier

🎙️ **An AI-powered voice agent for accessible respiratory and mental health screening.**

This platform transforms a simple phone call into a comprehensive health screening tool. It is designed specifically for rural accessibility, allowing users to interact with an AI voice agent in their native language to receive preliminary screenings for **Respiratory Diseases (COPD/Asthma)**, **Parkinson's Disease**, and **Depression**.

---

## 🇮🇳 Rural India Accessibility Suite (New)

We have implemented specific features to bridge the digital divide in rural India:

### 1. Zero-Cost "Missed Call" Service
*   **Problem**: Users often have no talk-time balance.
*   **Solution**: Users give a "Missed Call" to the helpline.
*   **Mechanism**: The server detects the incoming call, immediately **rejects** it (costing the user ₹0), and triggers an automatic outbound call back to the user's number.

### 2. ASHA Didi Mode (Community Health Worker)
*   **Problem**: Elderly or illiterate farmers may trust a local human component more than a machine.
*   **Solution**: Dedicated flow for ASHA workers (`Press 9`).
*   **Mechanism**: A worker calls once, enters a patient's mobile number, performs the screening, and is then seamlessly looped back to screen the next patient. Results are sent to the *patient's* phone, but managed by the worker.

### 3. Visual WhatsApp Health Cards
*   **Problem**: SMS text is hard to read; percentages are abstract.
*   **Solution**: Generates a visual "Traffic Light" (Red/Amber/Green) report card sent via WhatsApp.
*   **Tech**: Uses `Pillow` to dynamically generate an image with local language text and risk indicators.

### 4. Tele-Triage Bridge
*   **Problem**: Screening is useless without action.
*   **Solution**: If `Risk Level > High`, the call is **not hung up**.
*   **Mechanism**: The AI keeps the line open: *"Your symptoms concern me. Connecting you to a doctor now."* and bridges the call to the eSanjeevani government helpline.

### 5. "Storytelling" Persona
*   **Problem**: Clinical questions fail to get accurate answers.
*   **Solution**: Replaced checks like *"Do you have dyspnea?"* with *"Do you feel like a heavy stone is on your chest when you walk?"* utilizing culturally aware prompts.

---

## 🧠 Comprehensive Health Screening Models

The system uses a **Model Hub** architecture to run multiple diagnostic models on a single audio input:

1.  **Respiratory Health (PANNs Model)**
    *   **Architecture**: CNN6 (Convolutional Neural Network) trained on the ICBHI dataset.
    *   **Input**: Mel-spectrogram of cough/breath sound (16kHz).
    *   **Output**: Probability distribution for [Normal, Crackle, Wheeze].
    *   **Clinical Significance**: Crackles $\to$ Pneumonia/COPD; Wheezes $\to$ Asthma.

2.  **Parkinson's Disease Detection**
    *   **Architecture**: SVM (Support Vector Machine) Classifier.
    *   **Input**: 5-second sustained phonation ("Aaaah").
    *   **Features**: Extractions of Jitter (pitch perturbation), Shimmer (amplitude perturbation), HNR (Harmonics-to-Noise Ratio), and PPE (Pitch Period Entropy).
    *   **Output**: Risk probability.

3.  **Depression Screening**
    *   **Method**: Acoustic Biomarker Analysis.
    *   **Features**:
        *   *Prosody*: Pitch variability (monotonicity), Speech rate.
        *   *Energy*: Root Mean Square (RMS) energy levels.
        *   *Spectral*: Pause duration ratio.
    *   **Output**: Indicator analysis for "Flat Affect" and psychomotor retardation.

---

## ⚙️ Technical Deep Dive: How It Works

### The Call Flow Architecture

1.  **Ingestion**:
    *   User calls Twilio Number $\to$ `PROVISIONING_URL/india/voice/incoming`.
    *   Server greets in English/Hindi and asks for language selection.

2.  **Data Collection**:
    *   `POST /voice/handle-recording`: Audio is recorded (Linear PCM WAV).
    *   Questionnaire: Twilio `<Gather>` captures DTMF inputs for risk factors (smoking, chest pain).

3.  **Asynchronous Analysis Pipeline**:
    *   While the user answers the questionnaire, the **Background Task** begins:
        1.  Downloads audio from Twilio to local server.
        2.  **Model Hub** dispatches audio path to all 3 loaded models (Respiratory, PD, Depression).
        3.  Aggregates results into a `ComprehensiveHealthResult` object.
    
4.  **Real-Time Triage (The "Magic" Step)**:
    *   In `recording_complete_india`, the system checks `overall_risk_level`.
    *   **If Urgent**: Returns TwiML `<Dial>` to connect to a human doctor.
    *   **If Normal**: Returns TwiML `<Say>` with reassurance.

5.  **Report Generation**:
    *   `app/utils/image_generator.py` draws a `.png` based on the risk level.
    *   Twilio WhatsApp API sends this media to the user's endpoint.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- `ffmpeg` (for audio processing)
- Twilio Account (Voice & WhatsApp Sandbox)

### 1. Clone and Install
```bash
git clone https://github.com/YourUsername/PhoneCoughClassifier.git
cd PhoneCoughClassifier
pip install -r requirements.txt
# Ensure Pillow is installed for image generation
```

### 2. Configuration (`.env`)
```bash
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
DOCTOR_HELPLINE_NUMBER=+919999999999  # For Tele-Triage
ENABLE_WHATSAPP_REPORTS=True
```

### 3. Run the Server
```bash
python -m uvicorn app.main:app --reload
```

---

## 📊 API Endpoints

### Voice Webhooks
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/india/voice/incoming` | Main entry point. Handles language select. |
| `POST` | `/india/voice/missed-call` | Reject & Callback logic. |
| `POST` | `/india/voice/asha/menu` | Interface for Health Workers. |
| `POST` | `/india/voice/recording-complete` | Logic for analysis, triage, and reporting. |

### Testing & Models
| Endpoint | Description |
|----------|-------------|
| `/test/analyze-full` | Upload `.wav` file to test full model stack. |
| `/test/screening-models` | Check which models are loaded in memory. |

---

## 🏗️ Project Architecture

```
PhoneCoughClassifier/
├── app/
│   ├── api/                 # Webhooks (India, Twilio)
│   ├── ml/                  # Machine Learning Core
│   │   ├── model_hub.py          # Unified Interface
│   │   ├── voice_biomarkers.py   # Feature Extraction
│   ├── utils/
│   │   ├── image_generator.py    # WhatsApp Health Card Gen
│   ├── services/            # Twilio Interactions
│   └── database/            # SQLite Storage
├── external_models/         # Submodules (PANNs, etc.)
├── data/                    # Generated Health Cards
└── recordings/              # Audio storage
```

---

## 🔒 Privacy & Safety
- **No Diagnosis**: This tool provides a risk assessment/screening only.
- **HIPAA**: Audio is processed locally and can be configured to delete immediately after analysis.
- **Data Safety**: Patient IDs in ASHA mode are used only for reporting and not stored permanently on device.

🎙️ **An AI-powered voice agent for accessible respiratory and mental health screening.**

This platform allows users to call a phone number, interact with an AI voice agent in their native language, and receive preliminary health screenings for **Respiratory Diseases (COPD/Asthma)**, **Parkinson's Disease**, and **Depression** using advanced vocal biomarker analysis and prebuilt AI models.

## ✨ Key Features

### 🏥 Comprehensive Health Screening
Now integrates production-ready AI models to detect:

1.  **Respitory Health (PANNs Model)**
    *   **Source**: Integrated from [ilyassmoummad/scl_icbhi2017](https://github.com/ilyassmoummad/scl_icbhi2017)
    *   **Detects**: Crackles, Wheezes (Indicators for COPD, Asthma, Pneumonia)
    *   **Model**: CNN6 trained on ICBHI dataset.

2.  **Parkinson's Disease Detection**
    *   **Source**: Integrated from [AbderrezzakMrch/Parkinson-s-Disease-Voice-Detector](https://github.com/AbderrezzakMrch/Parkinson-s-Disease-Voice-Detector)
    *   **Detects**: Vocal tremors and biomarkers (Jitter, Shimmer, HNR, PPE).
    *   **Model**: Support Vector Machine (SVM) on voice features.

3.  **Depression Screening**
    *   **Source**: Integrated from [kykiefer/depression-detect](https://github.com/kykiefer/depression-detect)
    *   **Detects**: Flat affect, psychomotor retardation features (Pitch variability, Energy, Speaking Rate).
    *   **Method**: Acoustic feature extraction.

### 📞 Smart Voice Interface
- **Combined Screening Flow**: Users provide a single audio sample (cough + speech) to screen for ALL conditions simultaneously.
- **Background Processing**: Heavy AI analysis runs in the background while the user answers a breathing questionnaire, ensuring zero latency.
- **Multilingual Support**: Infrastructure for 10 Indian languages.

### 🏗️ Core Infrastructure
- **FastAPI Backend**: High-performance async Python web framework.
- **Model Hub**: Centralized manager that loads and coordinates multiple AI models (`app/ml/model_hub.py`).
- **Feature Extraction**: Professional-grade acoustic feature extraction using `librosa` and `opensmile` fallback.

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
```

### 2. External Models
The project relies on external model weights. The setup script (or manual download) retrieves these:
- **PANNs**: CNN6 weights (~22MB)
- **Parkinson's**: Pre-trained SVM model

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

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/twilio/voice/incoming` | Main entry point for phone calls |
| `POST` | `/test/analyze-full` | **Comprehensive Check**: Runs all 3 models on an uploaded file |
| `POST` | `/test/respiratory-screen` | Test PANNs respiratory model |
| `POST` | `/test/parkinsons-screen` | Test Parkinson's detector |
| `POST` | `/test/depression-screen` | Test Depression screener |
| `POST` | `/test/voice-biomarkers` | Extract raw biomarkers (Jitter, Shimmer, etc) |
| `GET`  | `/test/screening-models` | Check status of loaded models |

---

## 🏗️ Project Architecture

```
PhoneCoughClassifier/
├── app/
│   ├── api/                 # Webhooks & Test Endpoints
│   ├── ml/                  # Machine Learning Core
│   │   ├── model_hub.py          # Unified Interface for all models
│   │   ├── voice_biomarkers.py   # Feature Extraction
│   │   └── classifier.py         # Legacy Cough Classifier
│   ├── services/            # Twilio & IO Services
│   └── database/            # SQLite Storage
├── external_models/         # Cloned AI Repositories (PANNs, Parkinson's, etc.)
└── recordings/              # Audio storage
```

---

## 🔒 Privacy & Safety
- **No Diagnosis**: This tool provides a risk assessment/screening only, NOT a medical diagnosis.
- **Data Retention**: Audio files are processed and deleted.
- **Disclaimer**: Always consult a medical professional for serious symptoms.

