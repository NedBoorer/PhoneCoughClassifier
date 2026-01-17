# Medical Report Generation

## Overview

Generates concise, doctor-friendly medical reports from voice health screenings that can be:
- **Sent via SMS** (text message)
- **Printed** by users
- **Shown to doctors** during consultations

## Features

✅ **Brief format** - Bullet points, <1000 characters  
✅ **All screenings included** - TB, Respiratory, Parkinson's, Depression  
✅ **Clinical details** - Voice biomarkers, confidence scores  
✅ **Actionable recommendations** - Specific next steps (DOTS center, tests)  
✅ **Multi-language support** - Currently English, expandable  
✅ **Privacy-focused** - Phone numbers masked

## Report Types

### 1. Brief Medical Report (Default)
**Best for**: SMS to users, quick doctor reference

```
=== VOICE HEALTH SCREENING REPORT ===
Date: 2026-01-18 15:30
Patient ID: ***-***-3210
Overall Risk: MODERATE

FINDINGS:
• TB Screening: ⚠️ Moderate Risk
  - Confidence: 78%
  - Productive cough score: 0.85/1.0
• Respiratory: ⚠ Mild
  - Sound pattern: crackle
  - Suggests: COPD/pneumonia indicators
• Voice Tremor Analysis: ✓ Normal
  - Normal voice characteristics

CLINICAL RECOMMENDATION:
• Please visit nearest DOTS center...

RECOMMENDED FOLLOW-UP:
• Visit nearest DOTS center for sputum test/GeneXpert
• Chest X-ray recommended
```

**~800-1000 characters** (5-7 SMS segments)

### 2. Short Summary
**Best for**: Single SMS notification

```
VOICE HEALTH SCREEN (2026-01-18)
Risk: MODERATE
Findings: TB: moderate risk; Resp: crackle
Note: AI screening - needs clinical confirmation
```

**~200 characters** (2 SMS segments)

### 3. Detailed Printable Report
**Best for**: Printing, clinic records

```
============================================================
VOICE-BASED HEALTH SCREENING REPORT
============================================================

Report Date: January 18, 2026 at 03:30 PM

PATIENT INFORMATION:
Name: Rajesh Kumar
Age: 45 years
Gender: Male
Contact: ***-***-3210

------------------------------------------------------------
SCREENING RESULTS:
------------------------------------------------------------

TUBERCULOSIS (TB) SCREENING:
  Status: ⚠️ Moderate Risk
  Detection: POSITIVE
  Confidence: 78%
  Acoustic Features:
    - Productive cough: 0.85/1.0
    - Spectral centroid: 1450 Hz
  Recommendation: Visit DOTS center...

[... full details for all screenings ...]
```

**2000+ characters** (detailed, multi-page)

## Usage

### Automatic (After Every Screening)

Reports are **automatically sent** after phone call screenings:

1. User completes cough recording
2. System analyzes voice
3. Two SMS messages sent:
   - **Message 1**: User-friendly results
   - **Message 2**: Medical report for doctor

### Manual Request (API)

Users can request their latest report:

```bash
# Get latest report for a phone number
curl "http://localhost:8000/medical-report/latest/+919876543210?report_type=brief&send_sms=true"
```

**Parameters:**
- `report_type`: `brief` (default), `summary`, or `detailed`
- `send_sms`: `true` to send via SMS, `false` to just return

### Programmatic

```python
from app.services.twilio_service import format_medical_report
from app.ml.model_hub import get_model_hub

# After running analysis
hub = get_model_hub()
result = hub.run_full_analysis(audio_path)

# Generate report
report = format_medical_report(
    comprehensive_result=result,
    patient_phone="+919876543210",
    report_type="brief"  # or "summary", "detailed"
)

# Send via SMS
twilio_service.send_sms(phone_number, report)
```

## API Endpoints

### POST `/medical-report/generate`

Generate medical report from previous screening.

**Request:**
```json
{
  "phone_number": "+919876543210",
  "report_type": "brief",
  "send_sms": true
}
```

**Response:**
```json
{
  "success": true,
  "report": "=== VOICE HEALTH SCREENING REPORT ===\n...",
  "message": "Medical report generated and sent to +919876543210"
}
```

### GET `/medical-report/latest/{phone_number}`

Get latest report for a phone number.

```bash
GET /medical-report/latest/+919876543210?report_type=brief&send_sms=false
```

## What's Included

### All Screenings
- ✅ **Tuberculosis (TB)** - Primary focus for Indian farmers
- ✅ **Respiratory** - COPD, Asthma, Pneumonia indicators  
- ✅ **Parkinson's Disease** - Voice tremor analysis
- ✅ **Depression** - Speech pattern analysis

### Clinical Details
- Confidence scores (%)
- Severity levels
- Voice biomarkers (jitter, shimmer, HNR)
- Acoustic features (wetness score, spectral analysis)
- Sound patterns (crackle, wheeze)

### Recommendations
- Specific next steps
- Where to go (DOTS centers, clinics)
- What tests to request (GeneXpert, spirometry, etc.)
- Risk-based urgency

## Privacy & Security

- ✅ Phone numbers **masked** (***-***-3210)
- ✅ No patient names required
- ✅ Compliant with AI screening disclaimers
- ✅ Clear "NOT diagnostic" warnings

## Use Cases

### For Farmers
📱 "Show this message to doctor at PHC"
- Brief report texted after screening
- Can be screenshot and shown
- No internet needed at clinic

### For ASHA Workers
👩‍⚕️ Reference during home visits
- Printable report option
- Follow-up tracking
- Referral documentation

### For Doctors
👨‍⚕️ Quick clinical reference
- Key findings at a glance
- Voice biomarker data
- Screening vs. diagnostic clarity

## Example Output

Run the demo:
```bash
cd PhoneCoughClassifier
python examples/medical_report_demo.py
```

This shows:
1. Brief report (SMS format)
2. Short summary (single SMS)
3. Printable report (clinic format)

## SMS Cost Estimation

**Brief Report**: 5-7 SMS segments (~₹0.30-0.50)  
**Short Summary**: 2 SMS segments (~₹0.10-0.20)  
**Detailed Report**: Not recommended for SMS (use printable/web view)

## Integration Points

Reports are automatically sent in:
- ✅ `/twilio/recording-complete` (phone calls)
- ✅ `/voice-agent/recording-complete` (voice agent)
- ⚠️ WhatsApp (TODO: add medical report to health card)

## Future Enhancements

- [ ] Multi-language support (Hindi, Tamil, etc.)
- [ ] PDF generation with charts
- [ ] QR code linking to web report
- [ ] WhatsApp integration
- [ ] Email delivery option
- [ ] Historical report comparison

## Technical Details

**Files:**
- `app/utils/medical_report.py` - Report generators
- `app/services/twilio_service.py` - SMS integration
- `app/api/medical_report_endpoints.py` - API endpoints
- `examples/medical_report_demo.py` - Demo/examples

**Dependencies:**
- No additional packages needed (uses existing Twilio/FastAPI)

## Testing

```bash
# Run example
python examples/medical_report_demo.py

# Test API (with running server)
curl -X POST "http://localhost:8000/medical-report/generate" \
  -H "Content-Type: application/json" \
  -d '{"phone_number": "+919876543210", "report_type": "brief"}'
```

---

**Status**: ✅ Production Ready  
**Version**: 1.0  
**Last Updated**: 2026-01-18
