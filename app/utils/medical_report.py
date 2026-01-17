"""
Medical Report Generator
Generates concise, doctor-friendly reports from voice health screenings
"""
from datetime import datetime
from typing import Optional
from app.ml.model_hub import ComprehensiveHealthResult


def generate_medical_report(
    result: ComprehensiveHealthResult,
    patient_phone: str,
    language: str = "en",
    include_biomarkers: bool = True
) -> str:
    """
    Generate a brief, bullet-point medical report for doctors.
    
    Args:
        result: Comprehensive health screening result
        patient_phone: Patient phone number (masked for privacy)
        language: Report language (currently only 'en' supported)
        include_biomarkers: Include voice biomarker values
        
    Returns:
        Formatted medical report suitable for SMS/print
    """
    # Mask phone number for privacy (show last 4 digits)
    masked_phone = f"***-***-{patient_phone[-4:]}" if len(patient_phone) >= 4 else "****"
    
    # Generate report date
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    lines = [
        "=== VOICE HEALTH SCREENING REPORT ===",
        f"Date: {report_date}",
        f"Patient ID: {masked_phone}",
        f"Overall Risk: {result.overall_risk_level.upper()}",
        "",
        "FINDINGS:",
    ]
    
    # TB Screening (Priority #1)
    tb = result.screenings.get("tuberculosis")
    if tb:
        lines.append(f"• TB Screening: {_format_severity(tb.severity)}")
        if tb.detected:
            lines.append(f"  - Confidence: {int(tb.confidence * 100)}%")
            if tb.details.get("features"):
                wetness = tb.details["features"].get("wetness_score", 0)
                lines.append(f"  - Productive cough score: {wetness:.2f}/1.0")
        else:
            lines.append("  - No TB indicators detected")
    
    # Respiratory (COPD/Asthma)
    resp = result.screenings.get("respiratory")
    if resp:
        lines.append(f"• Respiratory: {_format_severity(resp.severity)}")
        if resp.detected:
            sound_class = resp.details.get("sound_class", "unknown")
            lines.append(f"  - Sound pattern: {sound_class}")
            lines.append(f"  - Confidence: {int(resp.confidence * 100)}%")
            
            # Clinical indicators
            if sound_class == "crackle":
                lines.append("  - Suggests: COPD/pneumonia indicators")
            elif sound_class == "wheeze":
                lines.append("  - Suggests: Asthma/bronchitis indicators")
            elif sound_class == "both":
                lines.append("  - Suggests: Multiple respiratory issues")
        else:
            lines.append("  - No abnormal respiratory sounds")
    
    # Parkinson's Disease
    pd = result.screenings.get("parkinsons")
    if pd:
        lines.append(f"• Voice Tremor Analysis: {_format_severity(pd.severity)}")
        if pd.detected:
            lines.append(f"  - Confidence: {int(pd.confidence * 100)}%")
            if include_biomarkers and pd.details.get("biomarkers"):
                biomarkers = pd.details["biomarkers"]
                if "jitter" in biomarkers:
                    lines.append(f"  - Jitter: {biomarkers['jitter']:.4f}%")
                if "shimmer" in biomarkers:
                    lines.append(f"  - Shimmer: {biomarkers['shimmer']:.4f}%")
                if "hnr" in biomarkers:
                    lines.append(f"  - HNR: {biomarkers['hnr']:.2f} dB")
        else:
            lines.append("  - Normal voice characteristics")
    
    # Depression Screening
    dep = result.screenings.get("depression")
    if dep:
        lines.append(f"• Speech Pattern Analysis: {_format_severity(dep.severity)}")
        if dep.detected:
            lines.append(f"  - Confidence: {int(dep.confidence * 100)}%")
            if dep.details.get("indicators"):
                indicators = dep.details["indicators"]
                # Filter clinical indicators
                clinical_indicators = [
                    ind for ind in indicators 
                    if not ind.endswith("_warning")
                ][:3]  # Max 3 indicators
                if clinical_indicators:
                    lines.append(f"  - Indicators: {', '.join(clinical_indicators)}")
        else:
            lines.append("  - Normal speech patterns")
    
    lines.extend([
        "",
        "CLINICAL RECOMMENDATION:",
        f"• {result.recommendation}",
        "",
        "NOTES:",
        "• AI-based screening tool (NOT diagnostic)",
        "• Requires clinical confirmation",
        "• Processing time: {:.1f}s".format(result.processing_time_ms / 1000),
    ])
    
    # Add specific follow-up actions
    follow_ups = _generate_follow_up_actions(result)
    if follow_ups:
        lines.append("")
        lines.append("RECOMMENDED FOLLOW-UP:")
        for action in follow_ups:
            lines.append(f"• {action}")
    
    lines.extend([
        "",
        "This report is for clinical reference only.",
        "Definitive diagnosis requires standard tests.",
    ])
    
    return "\n".join(lines)


def generate_short_medical_summary(
    result: ComprehensiveHealthResult,
    max_length: int = 500
) -> str:
    """
    Generate ultra-brief summary for SMS (160-500 chars).
    
    Args:
        result: Comprehensive health screening result
        max_length: Maximum character length
        
    Returns:
        Brief medical summary
    """
    findings = []
    
    # Check each screening
    tb = result.screenings.get("tuberculosis")
    if tb and tb.detected:
        findings.append(f"TB: {tb.severity} risk")
    
    resp = result.screenings.get("respiratory")
    if resp and resp.detected:
        sound = resp.details.get("sound_class", "abnormal")
        findings.append(f"Resp: {sound}")
    
    pd = result.screenings.get("parkinsons")
    if pd and pd.detected:
        findings.append(f"PD indicators: {pd.severity}")
    
    dep = result.screenings.get("depression")
    if dep and dep.detected:
        findings.append(f"Mood: {dep.severity}")
    
    if not findings:
        findings.append("All screenings normal")
    
    summary = [
        f"VOICE HEALTH SCREEN ({datetime.now().strftime('%Y-%m-%d')})",
        f"Risk: {result.overall_risk_level.upper()}",
        "Findings: " + "; ".join(findings),
        "Note: AI screening - needs clinical confirmation",
    ]
    
    text = "\n".join(summary)
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length-3] + "..."
    
    return text


def generate_printable_report(
    result: ComprehensiveHealthResult,
    patient_name: Optional[str] = None,
    patient_phone: Optional[str] = None,
    patient_age: Optional[int] = None,
    patient_gender: Optional[str] = None,
) -> str:
    """
    Generate a more detailed printable report with patient info.
    
    This can be sent as a text file attachment or displayed in a web view.
    """
    lines = [
        "=" * 60,
        "VOICE-BASED HEALTH SCREENING REPORT",
        "=" * 60,
        "",
        f"Report Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        "",
        "PATIENT INFORMATION:",
    ]
    
    if patient_name:
        lines.append(f"Name: {patient_name}")
    if patient_age:
        lines.append(f"Age: {patient_age} years")
    if patient_gender:
        lines.append(f"Gender: {patient_gender}")
    if patient_phone:
        masked = f"***-***-{patient_phone[-4:]}" if len(patient_phone) >= 4 else "****"
        lines.append(f"Contact: {masked}")
    
    lines.extend([
        "",
        "-" * 60,
        "SCREENING RESULTS:",
        "-" * 60,
        "",
        f"Overall Risk Level: {result.overall_risk_level.upper()}",
        "",
    ])
    
    # Detailed findings for each screening
    screenings_order = [
        ("tuberculosis", "TUBERCULOSIS (TB) SCREENING"),
        ("respiratory", "RESPIRATORY SOUND ANALYSIS"),
        ("parkinsons", "PARKINSON'S DISEASE INDICATORS"),
        ("depression", "DEPRESSION SCREENING (Speech Patterns)")
    ]
    
    for key, title in screenings_order:
        screening = result.screenings.get(key)
        if not screening:
            continue
        
        lines.append(f"{title}:")
        lines.append(f"  Status: {_format_severity(screening.severity)}")
        lines.append(f"  Detection: {'POSITIVE' if screening.detected else 'NEGATIVE'}")
        lines.append(f"  Confidence: {int(screening.confidence * 100)}%")
        
        # Add specific details
        if screening.detected and screening.details:
            if key == "tuberculosis":
                features = screening.details.get("features", {})
                if features:
                    lines.append("  Acoustic Features:")
                    if "wetness_score" in features:
                        lines.append(f"    - Productive cough: {features['wetness_score']:.2f}/1.0")
                    if "spectral_centroid_mean" in features:
                        lines.append(f"    - Spectral centroid: {features['spectral_centroid_mean']:.0f} Hz")
            
            elif key == "respiratory":
                sound_class = screening.details.get("sound_class")
                if sound_class:
                    lines.append(f"  Sound Pattern: {sound_class.upper()}")
                probs = screening.details.get("probabilities", {})
                if probs:
                    lines.append("  Sound Probabilities:")
                    for sound, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                        lines.append(f"    - {sound}: {int(prob * 100)}%")
            
            elif key == "parkinsons":
                biomarkers = screening.details.get("biomarkers", {})
                if biomarkers:
                    lines.append("  Voice Biomarkers:")
                    if "jitter" in biomarkers:
                        lines.append(f"    - Jitter (pitch perturbation): {biomarkers['jitter']:.4f}%")
                    if "shimmer" in biomarkers:
                        lines.append(f"    - Shimmer (amplitude variation): {biomarkers['shimmer']:.4f}%")
                    if "hnr" in biomarkers:
                        lines.append(f"    - HNR (harmonic-to-noise): {biomarkers['hnr']:.2f} dB")
            
            elif key == "depression":
                features = screening.details.get("features", {})
                if features:
                    lines.append("  Speech Features:")
                    if "pitch_std" in features:
                        lines.append(f"    - Pitch variability: {features['pitch_std']:.2f} Hz")
                    if "speaking_rate" in features:
                        lines.append(f"    - Speaking rate: {features['speaking_rate']:.2f} syllables/sec")
                    if "pause_ratio" in features:
                        lines.append(f"    - Pause ratio: {features['pause_ratio']:.2f}")
        
        lines.append(f"  Recommendation: {screening.recommendation}")
        lines.append("")
    
    lines.extend([
        "-" * 60,
        "OVERALL CLINICAL RECOMMENDATION:",
        "-" * 60,
        result.recommendation,
        "",
    ])
    
    # Follow-up actions
    follow_ups = _generate_follow_up_actions(result)
    if follow_ups:
        lines.append("RECOMMENDED NEXT STEPS:")
        for i, action in enumerate(follow_ups, 1):
            lines.append(f"  {i}. {action}")
        lines.append("")
    
    lines.extend([
        "-" * 60,
        "IMPORTANT DISCLAIMERS:",
        "-" * 60,
        "• This is an AI-based screening tool, NOT a diagnostic test",
        "• Results require confirmation by standard clinical methods",
        "• Positive TB screening requires sputum test/GeneXpert/CXR",
        "• Voice analysis is supplementary to clinical examination",
        f"• Processing completed in {result.processing_time_ms}ms",
        "",
        "=" * 60,
        "END OF REPORT",
        "=" * 60,
    ])
    
    return "\n".join(lines)


def _format_severity(severity: str) -> str:
    """Format severity level for display"""
    severity_map = {
        "normal": "✓ Normal",
        "low": "⚠ Low Risk",
        "low_risk": "⚠ Low Risk",
        "mild": "⚠ Mild",
        "moderate": "⚠️ Moderate Risk",
        "moderate_risk": "⚠️ Moderate Risk",
        "moderately_severe": "🔴 Moderately Severe",
        "high": "🔴 High Risk",
        "high_risk": "🔴 High Risk",
        "severe": "🚨 Severe",
        "urgent": "🚨 Urgent",
    }
    return severity_map.get(severity.lower(), severity.upper())


def _generate_follow_up_actions(result: ComprehensiveHealthResult) -> list:
    """Generate specific follow-up actions based on results"""
    actions = []
    
    # TB detected
    tb = result.screenings.get("tuberculosis")
    if tb and tb.detected and tb.severity in ["moderate_risk", "high_risk"]:
        actions.append("Visit nearest DOTS center for sputum test/GeneXpert")
        actions.append("Chest X-ray recommended")
    
    # Respiratory issues
    resp = result.screenings.get("respiratory")
    if resp and resp.detected:
        sound_class = resp.details.get("sound_class", "")
        if sound_class == "crackle":
            actions.append("Spirometry/lung function test recommended")
        elif sound_class == "wheeze":
            actions.append("Asthma evaluation with peak flow measurement")
        elif sound_class == "both":
            actions.append("Comprehensive pulmonary evaluation urgent")
    
    # Parkinson's
    pd = result.screenings.get("parkinsons")
    if pd and pd.detected and pd.severity in ["moderate", "high"]:
        actions.append("Neurological consultation for movement assessment")
        actions.append("Consider DAT scan if clinically indicated")
    
    # Depression
    dep = result.screenings.get("depression")
    if dep and dep.detected and dep.severity in ["moderate", "moderately_severe", "severe"]:
        actions.append("Mental health professional consultation")
        actions.append("PHQ-9 depression screening questionnaire")
    
    return actions
