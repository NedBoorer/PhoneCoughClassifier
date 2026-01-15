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

### 6. Kisan Manas (Farmer Mind) & Mandi Bol
*   **Problem**: Farmers are reluctant to seek help for mental health due to stigma ("log kya kahenge").
*   **Solution**: **Passive Screening** hidden inside a routine utility service (Market Prices).
*   **Mechanism**: 
    1.  Farmer calls "Mandi Bol" to check onion/tomato prices.
    2.  System asks: *"To give you the best price, describe your crop quality for 10 seconds."*
    3.  While the farmer speaks, the **Depression Classifier** analyzes voice biomarkers (monotone, low energy, jitter).
    4.  **Intervention**: If distress is detected, the call gently pivots: *"I noticed you sound tired... Our Kisan Mitra counselor is here to listen."*
    5.  User can press 1 to connect to a counselor immediately.

---

## 📱 User Journeys & Workflows

### 1. The "Missed Call" Journey (Zero Cost)
*   **User Action**: Farmer dials the toll-free number and hangs up after one ring.
*   **System Response**: Server rejects the call (busy signal) to prevent any charge.
*   **Callback**: Server immediately triggers an outbound call back to the farmer's number.
*   **Result**: The health screening happens entirely at the server's expense.

### 2. The Comprehensive Health Screen (Active)
1.  **Greeting**: "Namaste. I am your health friend..."
2.  **Language Select**: User picks Hindi (`Press 2`).
3.  **Symptom Check**: "Do you have chest pain?" (Yes/No via DTMF).
4.  **Unified Recording**: "Please cough and then count to 10." (Captures Cough + Voice).
    *   *Background Magic*: Audio is processed in parallel.
    *   Cough Segment $\to$ **PANNs Model** (Respiratory check).
    *   Voice Segment $\to$ **SVM** (Parkinson's check) + **Biomarker Analysis** (Depression check).
5.  **Triage**:
    *   *Normal*: "You seem healthy." $\to$ **WhatsApp Report** sent with Green Card.
    *   *High Risk*: "Please hold, connecting you to a doctor." $\to$ Call bridged effectively to `DOCTOR_HELPLINE_NUMBER`.

### 3. ASHA Worker Mode (Community Screening)
Designed for a health worker visiting a village with one smartphone.
1.  **Login**: Worker calls and presses `9` at the language menu.
2.  **Patient Identification**: Worker enters the **Patient's Mobile Number**.
3.  **Screening**: Worker hands phone to patient for the recording phase.
4.  **Reporting**: Result is stored against the *Patient's* ID. WhatsApp report is sent to the *Patient's* number (if available).
5.  **rapid Loop**: Call returns to the main menu immediately, ready to screen the next patient in the queue.

### 4. "Mandi Bol" (Passive Screening)
1.  **Intent**: User calls for market prices, *not* health.
2.  **Action**: Calls "Mandi Bol" line.
3.  **Interaction**: "To get the best price for Onions, describe your crop quality."
4.  **Passive Scan**: System silently analyzes the *prosody* and *energy* of the voice during the crop description.
5.  **Intervention**: If (and only if) signs of severe distress are detected, the system gently interrupts: *"I noticed you sound tired... Our Kisan Mitra counselor is here to listen."*


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
| `POST` | `/india/voice/market/menu` | Mandi Bol: Market Prices & Passive Screen |

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



