# 🌾 Phone Cough Classifier: Farmer Health Edition

## Complete Guide with Customer Journeys & Implementation Details

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Key Features for Rural Farmers](#key-features)
3. [Customer Journeys](#customer-journeys)
4. [Technical Architecture](#technical-architecture)
5. [Setup & Deployment](#setup-deployment)
6. [API Reference](#api-reference)
7. [Suggested Follow-Ups](#suggested-follow-ups)
8. [Impact Metrics](#impact-metrics)

---

## 🎯 Overview

The **Phone Cough Classifier: Farmer Health Edition** is an AI-powered voice health screening system specifically optimized for **rural Indian farmers**. It addresses unique challenges:

- **Low literacy**: Visual (traffic light) + audio-first design
- **Feature phones**: No smartphone required, DTMF + voice input
- **Cost sensitivity**: Missed call callback (₹0 cost to user)
- **Occupational hazards**: Pesticide & dust exposure screening
- **Farming seasons**: Respects farmer availability (harvest/sowing)
- **Family-centric**: Screen entire household in one session
- **2G/3G networks**: Optimized images load in 30 seconds

---

## 🌟 Key Features for Rural Farmers

### 1. **Zero-Cost Missed Call Service**
Farmers call → System rejects → Automatic callback (farmer pays ₹0)

### 2. **Multilingual Support (10 Indian Languages)**
- English, Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, Kannada, Malayalam, Punjabi
- Full voice prompts in all languages
- WhatsApp reports in local language

### 3. **Occupational Health Screening**
Special questions for farmers:
- "Do you use pesticides?" → Escalates respiratory symptoms to URGENT
- "Do you work with grain/dust?" → Screens for Farmer's Lung disease
- "What farming season?" → Adjusts follow-up timing

### 4. **Family Health Tracking**
- Link multiple family members to one household
- ASHA workers can screen entire families
- Family health dashboard shows collective risk

### 5. **Seasonal Follow-Up Calendar**
Respects farming cycle:
- **Off-season (Feb-May)**: Follow up within 3 days (farmers have time)
- **Sowing (Jun-Jul)**: Follow up within 14 days (moderately busy)
- **Growing (Aug-Oct)**: Normal follow-up (7 days)
- **Harvest (Nov-Jan)**: Follow up within 21 days (very busy) → SMS instead of call

### 6. **ASHA Worker Mode**
Community health workers can:
- Screen multiple families per day
- Get daily summary SMS
- Track villages and high-risk cases
- Link family members to households

### 7. **Low-Bandwidth Optimized**
- WhatsApp images: 40KB (vs 200KB) → Loads in 30 sec on 2G
- JPEG 60% quality (vs PNG)
- 400x500 pixels (vs 800x1000)
- Traffic light color coding (Red/Amber/Green) for low literacy

---

## 🚶 Customer Journeys

### Journey 1: Ramesh, the Farmer (Solo Screening)

**Profile:**
- 45-year-old farmer in Maharashtra
- Uses pesticides during cotton season
- Has chronic cough
- Feature phone only (no internet)
- Speaks Marathi

**Journey:**

1. **Call Initiation (Missed Call)**
   - Ramesh calls the health number
   - System rejects (he pays ₹0)
   - System calls him back within 30 seconds

2. **Language Selection**
   ```
   Voice: "For English, press 1. मराठीसाठी 2 दाबा."
   Ramesh: [Presses 2]
   ```

3. **Occupation Screening** (NEW)
   ```
   Voice: "तुम्ही शेतकरी आहात का? होय साठी 1, नाही साठी 2 दाबा."
   Ramesh: [Presses 1 - Yes, I'm a farmer]
   ```

4. **Pesticide Exposure** (NEW)
   ```
   Voice: "तुम्ही किडेनाशक वापरता का? होय साठी 1, नाही साठी 2 दाबा."
   Ramesh: [Presses 1 - Yes, I use pesticides]
   ```

5. **Farming Season** (NEW)
   ```
   Voice: "आता कोणता हंगाम आहे? पेरणीसाठी 1, वाढीसाठी 2, कापणीसाठी 3."
   Ramesh: [Presses 3 - Harvest season]
   ```

6. **Cough Recording**
   ```
   Voice: "बीप नंतर खोकला."
   [Beep]
   Ramesh: [Coughs 3 times]
   ```

7. **AI Analysis** (Background - 5 seconds)
   - Classification: "Chronic cough"
   - Pesticide exposure: YES → Escalate to HIGH risk
   - Dust exposure: YES → Add Farmer's Lung warning

8. **Immediate Voice Result**
   ```
   Voice: "तुमची खांसी जुनाट आहे. किडेनाशकांचा वापर धोकादायक आहे.
          आज डॉक्टरांना भेटा. हे किडेनाशक विषबाधा असू शकते."
   ```

9. **WhatsApp Health Card** (30 seconds later)
   - Traffic light: 🔴 RED (High Risk)
   - Image loads in 30 seconds (400x500, 40KB)
   - Text in Marathi:
     ```
     ⚠️ उच्च धोका
     किडेनाशक + जुनाट खोकला
     तातडीने डॉक्टरांना भेटा
     ```

10. **Follow-Up Scheduling** (Smart)
    - Normal: 24 hours
    - **BUT** Harvest season detected → Delayed to 21 days
    - SMS sent: "कापणी व्यस्त आहे. हंगामानंतर चेक करू."

**Outcome:**
- Ramesh screened in 2 minutes
- High-risk detected early (pesticide + chronic cough)
- Follow-up respects his harvest schedule
- Cost to Ramesh: ₹0
- Family linked for future screenings

---

### Journey 2: Sunita, the ASHA Worker (Family Screening)

**Profile:**
- ASHA worker in Telangana village
- Covers 15 families
- Uses basic smartphone
- Screens 3-5 families per day

**Journey:**

1. **ASHA Login**
   - Calls ASHA worker number
   - Enters ASHA ID: 2734

2. **Daily Screening Start**
   ```
   Voice: "Welcome ASHA Sunita. Press 1 for Family Screening, 2 for Daily Summary."
   Sunita: [Presses 1]
   ```

3. **Family Selection**
   ```
   Voice: "Enter family primary contact number."
   Sunita: [Enters: 9876543210 - Gopal's family]
   ```

4. **Family Size**
   ```
   Voice: "How many family members to screen? Press 1-9."
   Sunita: [Presses 4 - Gopal, wife, 2 children]
   ```

5. **Member 1: Gopal (Father)**
   ```
   Voice: "Recording for family member 1. Hand phone to them."
   [Gopal coughs]
   → Result: NORMAL ✓
   ```

6. **Member 2: Lakshmi (Mother)**
   ```
   Voice: "Recording for family member 2."
   [Lakshmi coughs]
   → Occupation: Farmer
   → Dust exposure: YES
   → Result: MODERATE ⚠️ (Dust + Dry cough)
   ```

7. **Member 3: Ravi (Son, 12)**
   ```
   Voice: "Recording for family member 3."
   [Ravi coughs]
   → Result: NORMAL ✓
   ```

8. **Member 4: Priya (Daughter, 8)**
   ```
   Voice: "Recording for family member 4."
   [Priya coughs]
   → Result: NORMAL ✓
   ```

9. **Family Summary SMS** (Sent to Gopal + Sunita)
   ```
   🏠 కుటుంబ ఆరోగ్య నివేదిక
   కుటుంబం: 9876543210
   గ్రామం: Khammam

   సభ్యులు: 4

   ⚠️ పర్యవేక్షించండి:
   🟡 Lakshmi: పొడి దగ్గు (Dust exposure)

   ✓ సాధారణం:
   🟢 Gopal: సాధారణం
   🟢 Ravi: సాధారణం
   🟢 Priya: సాధారణం

   ఆరోగ్య హెల్ప్‌లైన్: 108
   ```

10. **End of Day Summary** (6 PM)
    ```
    SMS to Sunita:
    "📊 Today's Summary - ASHA 2734
     ✓ 4 families screened (17 members)
     🔴 2 high-risk: Ramesh, Suresh
     🟡 3 moderate-risk
     🟢 12 normal
     Village: Khammam"
    ```

**Outcome:**
- 1 family (4 members) screened in 8 minutes
- All linked to Gopal's family group
- Lakshmi flagged for follow-up (dust exposure)
- ASHA gets daily performance summary
- Village health data tracked

---

### Journey 3: Mohan, the Farm Worker (Voice-Only Mode)

**Profile:**
- Farm laborer in Punjab
- Old Nokia phone with broken keypad
- Cannot press numbers (DTMF broken)
- Speaks Punjabi only

**Journey:**

1. **Call Start**
   ```
   Voice (Punjabi): "ਸੁਆਗਤ ਹੈ. ਕੀ ਤੁਸੀਂ ਕਿਸਾਨ ਹੋ?
                    ਹਾਂ ਕਹੋ ਜਾਂ 1 ਦਬਾਓ. ਨਾਂ ਕਹੋ ਜਾਂ 2 ਦਬਾਓ."
   Mohan: "ਹਾਂ" (Says "Haan" - Yes)
   ```
   → System detects speech: "haan" → Farmer = YES

2. **Pesticide Question**
   ```
   Voice: "ਕੀ ਤੁਸੀਂ ਕੀੜੇਮਾਰ ਵਰਤਦੇ ਹੋ? ਹਾਂ ਜਾਂ ਨਾਂ ਕਹੋ."
   Mohan: "ਨਹੀਂ" (Says "Nahi" - No)
   ```
   → Speech detected: "nahi" → Pesticide = NO

3. **Dust Exposure**
   ```
   Voice: "ਕੀ ਤੁਸੀਂ ਅਨਾਜ ਨਾਲ ਕੰਮ ਕਰਦੇ ਹੋ? ਹਾਂ ਜਾਂ ਨਾਂ ਕਹੋ."
   Mohan: "ਹਾਂ" (Says "Haan" - Yes)
   ```
   → Dust exposure = YES

4. **Cough Recording**
   [Mohan coughs]

5. **Result**
   - Dry cough + Dust exposure → MODERATE risk
   - Farmer's Lung warning added
   - SMS in Punjabi: "⚠️ ਧੂੜ + ਖੰਘ = ਮਾਸਕ ਪਹਿਨੋ. ਡਾਕਟਰ ਨੂੰ ਮਿਲੋ."

**Outcome:**
- **Voice-only mode works!** (No keypad needed)
- Speech recognition in Punjabi successful
- Farmer screened despite broken phone
- Low-literacy farmer understood traffic light image

---

## 🏗️ Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     PHONE COUGH CLASSIFIER                  │
│                    (Farmer Health Edition)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │         Twilio Voice Gateway             │
        │  - Missed call detection                 │
        │  - Auto-callback                         │
        │  - DTMF + Speech recognition             │
        │  - Multi-language TTS                    │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │        FastAPI Backend (Async)           │
        │  - india_webhooks.py (Farmer flows)      │
        │  - family_endpoints.py (Dashboard)       │
        │  - admin_tasks.py (Follow-ups)           │
        └─────────────────────────────────────────┘
                              │
                ┌─────────────┴──────────────┐
                ▼                            ▼
    ┌───────────────────┐      ┌────────────────────────┐
    │  ML Model Hub     │      │  SQLite Database       │
    │  ✓ Respiratory    │      │  ✓ CallRecords         │
    │  ✓ Parkinson's    │      │  ✓ FamilyGroups        │
    │  ✓ Depression     │      │  ✓ PatientInfo         │
    │  ✓ Occupational   │      │  ✓ Classifications     │
    │    Risk Logic     │      │  ✓ ASHA assignments    │
    └───────────────────┘      └────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │     Output Channels                      │
        │  - WhatsApp (2G-optimized images)        │
        │  - SMS (multilingual reports)            │
        │  - Voice (immediate feedback)            │
        └─────────────────────────────────────────┘
```

### Key Technologies

- **Backend**: FastAPI (async), Python 3.9+
- **Voice**: Twilio Voice API, Polly TTS
- **ML**: PyTorch (PANNs), scikit-learn, librosa
- **Database**: SQLite (AsyncIO), SQLAlchemy 2.0
- **Images**: Pillow (optimized JPEG)
- **Messaging**: Twilio SMS/WhatsApp

### Database Schema

```sql
-- ENHANCED FOR FARMERS
CREATE TABLE family_groups (
    id INTEGER PRIMARY KEY,
    primary_contact_number VARCHAR(20),
    village VARCHAR(100),
    district VARCHAR(100),
    household_size INTEGER,
    has_high_risk_member BOOLEAN,
    asha_worker_id VARCHAR(50),
    last_screening_date DATETIME
);

CREATE TABLE patient_info (
    id INTEGER PRIMARY KEY,
    call_id INTEGER FOREIGN KEY,
    family_group_id INTEGER FOREIGN KEY,  -- NEW

    -- FARMER FIELDS (NEW)
    occupation VARCHAR(30),               -- farmer, farm_worker
    pesticide_exposure BOOLEAN,           -- Occupational hazard
    dust_exposure BOOLEAN,                -- Grain/hay/dust
    farming_season VARCHAR(20)            -- sowing, harvest, off_season
);

CREATE TABLE call_records (
    id INTEGER PRIMARY KEY,
    caller_number VARCHAR(20),
    language VARCHAR(10),

    -- FOLLOW-UP TRACKING
    followup_status VARCHAR(20),          -- pending, scheduled, sms_sent
    followup_scheduled_at DATETIME,
    followup_attempted_at DATETIME
);
```

---

## 🛠️ Setup & Deployment

### Prerequisites

```bash
# System requirements
- Python 3.9+
- ffmpeg (for audio processing)
- SQLite 3.35+

# For production
- Twilio account (Voice, SMS, WhatsApp)
- Domain with HTTPS (for webhooks)
- Server with 2GB RAM minimum
```

### Installation

```bash
# 1. Clone repository
git clone <repo-url>
cd PhoneCoughClassifier

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download ML model weights
python scripts/download_models.py

# 5. Setup environment variables
cp .env.example .env
# Edit .env with your credentials:
#   TWILIO_ACCOUNT_SID=...
#   TWILIO_AUTH_TOKEN=...
#   TWILIO_PHONE_NUMBER=+91...
#   BASE_URL=https://yourdomain.com

# 6. Initialize database
python -c "from app.database.database import init_db; import asyncio; asyncio.run(init_db())"

# 7. Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment

```bash
# Using Docker (Recommended)
docker build -t phone-cough-classifier .
docker run -p 8000:8000 --env-file .env phone-cough-classifier

# Or using systemd
sudo cp deployment/phone-classifier.service /etc/systemd/system/
sudo systemctl enable phone-classifier
sudo systemctl start phone-classifier

# Setup Nginx reverse proxy
sudo cp deployment/nginx.conf /etc/nginx/sites-available/phone-classifier
sudo ln -s /etc/nginx/sites-available/phone-classifier /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

### Twilio Configuration

1. **Buy Phone Number**: Indian number (+91)
2. **Configure Webhook URLs**:
   ```
   Voice Incoming: https://yourdomain.com/india/voice/incoming
   SMS Incoming: https://yourdomain.com/india/sms/incoming
   ```
3. **Enable WhatsApp Sandbox** (Testing) or Request Production Access
4. **Configure Missed Call**:
   - Set Voice URL to return immediate <Reject/>
   - Use Status Callback to trigger callback

---

## 📞 API Reference

### Core Endpoints

#### 1. **Farmer Screening Flow**

```http
POST /india/voice/incoming
Content-Type: application/x-www-form-urlencoded

From=+919876543210&CallSid=CA123...
```

**Response**: TwiML with language selection

#### 2. **Family Status Dashboard**

```http
GET /family/status/+919876543210
Authorization: Bearer <api_key>
```

**Response**:
```json
{
  "family_group_id": 42,
  "primary_contact": "+919876543210",
  "household_size": 4,
  "members_screened": 4,
  "high_risk_members": 1,
  "village": "Khammam",
  "asha_worker_assigned": true,
  "members": [
    {
      "patient_id": 101,
      "calling_for": "self",
      "age_group": "40_60",
      "last_screening": "2026-01-15T10:30:00Z",
      "classification": "chronic",
      "severity": "high",
      "occupation": "farmer",
      "pesticide_exposure": true
    },
    // ... more members
  ]
}
```

#### 3. **ASHA Dashboard**

```http
GET /family/asha-dashboard/ASHA2734
```

**Response**:
```json
{
  "asha_worker_id": "ASHA2734",
  "total_families": 15,
  "total_members": 63,
  "families_with_high_risk": 3,
  "needs_attention": [
    {
      "family_id": 42,
      "contact": "+919876543210",
      "village": "Khammam",
      "reason": "High-risk member",
      "last_screening": "2026-01-14"
    }
  ]
}
```

#### 4. **Trigger Seasonal Follow-Ups**

```http
POST /admin/trigger-followups
Authorization: Bearer <admin_key>
```

**Response**:
```json
{
  "status": "success",
  "triggered_calls": 5,
  "sms_reminders": 12,
  "current_season": "harvest",
  "total_processed": 17
}
```

**Logic**:
- Urgent cases: Call within 24h (ignores season)
- Farmers in harvest: SMS reminder (no call)
- Other farmers: Seasonal delay applied

---

## 🚀 Suggested Follow-Ups

### Phase 1: Pilot Testing (Month 1-2)

**Objective**: Validate farmer-specific features in 1-2 villages

**Tasks**:
1. **Select Pilot Villages**
   - Choose 2 villages in different states (e.g., Maharashtra, Telangana)
   - 50-100 farmer households each
   - Good ASHA worker coverage

2. **ASHA Training**
   - 2-day workshop for 5-10 ASHA workers
   - Hands-on family screening practice
   - Daily reporting workflows

3. **Metrics to Track**:
   - Total calls received
   - Farmers screened (% of target population)
   - Family screening adoption (% screening >1 member)
   - High-risk cases detected
   - Follow-up completion rate
   - WhatsApp image load success (2G/3G)

4. **Weekly Reviews**:
   - ASHA worker feedback sessions
   - Adjust voice prompts based on farmer feedback
   - Review occupational risk escalations (false positives?)

**Expected Outcomes**:
- 60%+ farmer participation
- 30%+ family screening adoption
- Identify 5-10 high-risk cases (pesticide exposure)
- Validate seasonal follow-up logic

---

### Phase 2: Scale to District (Month 3-4)

**Objective**: Expand to 20-30 villages in 1 district

**Tasks**:
1. **Recruit & Train ASHA Workers**
   - 30-50 ASHA workers
   - District health officer coordination

2. **Infrastructure**:
   - Ensure server capacity (500+ calls/day)
   - Monitor WhatsApp delivery rates
   - Set up district dashboard

3. **Government Integration**:
   - Link to eSanjeevani tele-consultation
   - Share data with District Health Department
   - Integrate with National Health Mission (NHM) reporting

4. **Add Crop-Specific Warnings**:
   - Cotton farmers → Pesticide warnings
   - Rice farmers → Water-borne illness alerts
   - Wheat farmers → Dust exposure monitoring

**Expected Outcomes**:
- 1,000-2,000 farmers screened
- 200+ families linked
- 50+ high-risk cases detected
- Government partnership established

---

### Phase 3: State-Level Rollout (Month 5-12)

**Objective**: Cover entire state (1M+ farmers)

**Tasks**:
1. **Multi-District Coordination**
   - Expand to 10-20 districts
   - State health department MoU
   - Budget allocation for ASHA incentives

2. **Advanced Features**:
   - **Crop Calendar Integration**: Adjust follow-ups based on local crop cycles
   - **Weather Alerts**: Heat stroke warnings during summer
   - **Supply Chain**: Link to nearest clinic with medications
   - **Predictive Analytics**: Identify villages at risk for epidemics

3. **Clinical Validation**:
   - Partner with medical colleges
   - Validate AI predictions vs. doctor diagnosis
   - Publish research paper on occupational health screening

4. **Policy Advocacy**:
   - Present to National Health Mission
   - Integrate into Ayushman Bharat Digital Health
   - Seek funding from Ministry of Health & Family Welfare

**Expected Outcomes**:
- 100,000+ farmers screened in Year 1
- 10,000+ families tracked
- 500+ high-risk cases detected early
- Government adoption and funding

---

### Phase 4: National Expansion (Year 2+)

**Objective**: Multi-state coverage (10M+ farmers)

**Tasks**:
1. **Geographic Expansion**
   - Replicate in 10 states
   - Localize for regional crops (cotton, rice, wheat, sugarcane)

2. **Advanced AI Models**:
   - Fine-tune on Indian farmer data (10,000+ samples)
   - Add crop-specific respiratory models
   - Integrate air quality data (pollution + pesticide)

3. **Ecosystem Integration**:
   - Link to AgriTech platforms (Kisan Call Center)
   - Integrate with crop insurance claims
   - Partner with pharma for medication distribution

4. **Research & Publications**:
   - Longitudinal study: Farmer health over 5 years
   - Impact assessment: Lives saved, early detections
   - Policy paper: Occupational health standards for farmers

**Expected Outcomes**:
- 1M+ farmers screened annually
- 100,000+ families tracked
- Influence national policy on farmer occupational health
- Save 1,000+ lives per year (early detection of serious conditions)

---

## 📊 Impact Metrics

### Health Outcomes

| Metric | Target (Year 1) | Measurement Method |
|--------|-----------------|-------------------|
| Farmers screened | 100,000+ | Call records |
| High-risk cases detected | 500+ | Classification = "urgent" or "high" |
| Early pesticide poisoning detections | 50+ | Pesticide exposure + respiratory |
| Farmer's Lung cases identified | 30+ | Dust exposure + chronic cough |
| Follow-up completion rate | 70%+ | Followup_status = "completed" |
| Clinical validation accuracy | 85%+ | AI prediction vs. doctor diagnosis |

### Adoption Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Family screening adoption | 30%+ | % calls with family_group_id |
| ASHA worker adoption | 80%+ | % ASHA workers using system weekly |
| Repeat usage | 40%+ | % farmers screening >1 time |
| WhatsApp image success rate | 90%+ | % images delivered successfully |
| 2G network compatibility | 95%+ | % images loaded in <60 sec |

### Economic Impact

| Metric | Calculation |
|--------|------------|
| **Cost per screening** | ₹10-15 (vs ₹500 in-person) |
| **Farmer cost** | ₹0 (missed call model) |
| **Lives saved** | 50-100 per year (early detection) |
| **Productivity gain** | 10,000 work-days saved (avoiding severe illness) |
| **ROI** | 10:1 (healthcare costs avoided vs. system cost) |

---

## 🌾 Farmer-Specific Success Stories (Projected)

### Story 1: Ramesh's Pesticide Poisoning Detection
- **Before**: Would have continued spraying, symptoms worsened → Hospital (₹50,000 cost)
- **After**: Detected early, stopped pesticide use, simple treatment (₹2,000 cost)
- **Impact**: ₹48,000 saved, 3 weeks work-days saved

### Story 2: Sunita's Village Screening
- **Before**: ASHA worker visits 2-3 homes/day, manual records
- **After**: Screens 5 families/day (20+ people), digital tracking
- **Impact**: 2.5x efficiency, better follow-up compliance

### Story 3: Lakshmi's Farmer's Lung
- **Before**: Chronic cough ignored → Permanent lung damage
- **After**: Dust mask provided, symptoms improved in 2 months
- **Impact**: Work capacity restored, long-term disability avoided

---

## 🔄 Integration Opportunities

### 1. **AgriTech Platforms**
- **Kisan Call Center**: Add health screening to ag advice calls
- **IFFCO Kisan**: Integrate with farmer app
- **DeHaat**: Bundle with ag input purchases

### 2. **Government Schemes**
- **Ayushman Bharat**: Link to health ID
- **PM-KISAN**: Health screening with subsidy transfer
- **eNAM**: Market + health checkup combo

### 3. **Insurance**
- **Pradhan Mantri Fasal Bima Yojana**: Health + crop insurance
- **Premium discount**: For farmers who screen regularly

### 4. **Telemedicine**
- **eSanjeevani**: Direct transfer for high-risk cases
- **Practo/mFine**: Paid consultation for moderate cases

---

## 📚 Additional Resources

### For Developers
- [API Documentation](/docs)
- [ML Model Architecture](MODELS.md)
- [Database Schema](DATABASE.md)
- [Deployment Guide](DEPLOYMENT.md)

### For ASHA Workers
- [Training Manual (Hindi)](training/asha_training_hindi.pdf)
- [Quick Reference Card](training/quick_reference.pdf)
- [Troubleshooting Guide](training/troubleshooting.pdf)

### For Policy Makers
- [Impact Assessment Report](reports/impact_assessment.pdf)
- [Cost-Benefit Analysis](reports/cost_benefit.pdf)
- [Scalability Study](reports/scalability.pdf)

---

## 🙏 Acknowledgements

This system is designed to serve **rural Indian farmers**, the backbone of our nation's food security. Special thanks to:

- **ASHA Workers**: Frontline health heroes who bridge the last mile
- **Farmers**: Who feed the nation despite facing immense challenges
- **Government Health Departments**: For partnerships and support
- **Open Source Community**: For ML models and tools

---

## 📞 Contact & Support

- **Technical Issues**: support@phonecoughclassifier.org
- **Partnership Inquiries**: partnerships@phonecoughclassifier.org
- **ASHA Training**: training@phonecoughclassifier.org
- **Farmer Helpline**: 1800-XXX-XXXX (Toll-Free)

---

## 📄 License

This project is licensed under the MIT License with a special clause: **Use for social good is encouraged and free. Commercial use requires attribution and revenue sharing with farmer cooperatives.**

---

**Built with ❤️ for Rural India** 🇮🇳 🌾

*"Technology that reaches the last mile is technology that matters."*
