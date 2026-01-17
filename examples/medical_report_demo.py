"""
Medical Report Generation Examples

This demonstrates the different types of medical reports that can be generated.
"""
from app.ml.model_hub import ComprehensiveHealthResult, ScreeningResult
from app.utils.medical_report import (
    generate_medical_report,
    generate_short_medical_summary,
    generate_printable_report
)


def create_sample_result():
    """Create a sample screening result for testing"""
    return ComprehensiveHealthResult(
        primary_concern="tuberculosis",
        overall_risk_level="moderate",
        screenings={
            "tuberculosis": ScreeningResult(
                disease="tuberculosis",
                detected=True,
                confidence=0.78,
                severity="moderate_risk",
                details={
                    "features": {
                        "wetness_score": 0.85,
                        "spectral_centroid_mean": 1450.5,
                        "duration": 3.2
                    }
                },
                recommendation="Please visit a DOTS center for sputum test/GeneXpert confirmation."
            ),
            "respiratory": ScreeningResult(
                disease="respiratory",
                detected=True,
                confidence=0.65,
                severity="mild",
                details={
                    "sound_class": "crackle",
                    "probabilities": {
                        "normal": 0.35,
                        "crackle": 0.65,
                        "wheeze": 0.0,
                        "both": 0.0
                    }
                },
                recommendation="Crackles detected - consider lung function testing."
            ),
            "parkinsons": ScreeningResult(
                disease="parkinsons",
                detected=False,
                confidence=0.88,
                severity="normal",
                details={
                    "biomarkers": {
                        "jitter": 0.0045,
                        "shimmer": 0.0312,
                        "hnr": 22.5
                    }
                },
                recommendation="No significant Parkinson's voice indicators detected."
            ),
        },
        voice_biomarkers={
            "wetness_score": 0.85,
            "jitter": 0.0045,
            "shimmer": 0.0312,
        },
        processing_time_ms=2850,
        recommendation="TB screening shows moderate risk. Please visit nearest DOTS center for confirmatory testing (free of cost). Early detection ensures successful treatment."
    )


def example_brief_report():
    """Example: Brief medical report (SMS-friendly, ~800 chars)"""
    result = create_sample_result()
    
    report = generate_medical_report(
        result=result,
        patient_phone="+919876543210",
        language="en",
        include_biomarkers=True
    )
    
    print("=" * 60)
    print("BRIEF MEDICAL REPORT (For SMS/Doctor)")
    print("=" * 60)
    print(report)
    print("\n")
    print(f"Character count: {len(report)}")
    print(f"SMS segments needed: {(len(report) // 160) + 1}")


def example_short_summary():
    """Example: Ultra-brief summary (~200 chars)"""
    result = create_sample_result()
    
    summary = generate_short_medical_summary(
        result=result,
        max_length=500
    )
    
    print("=" * 60)
    print("SHORT SUMMARY (Single SMS)")
    print("=" * 60)
    print(summary)
    print("\n")
    print(f"Character count: {len(summary)}")


def example_printable_report():
    """Example: Detailed printable report"""
    result = create_sample_result()
    
    report = generate_printable_report(
        result=result,
        patient_name="Rajesh Kumar",
        patient_phone="+919876543210",
        patient_age=45,
        patient_gender="Male"
    )
    
    print("=" * 60)
    print("PRINTABLE REPORT (For Doctor/Clinic)")
    print("=" * 60)
    print(report)
    print("\n")
    print(f"Character count: {len(report)}")
    print(f"Lines: {len(report.split(chr(10)))}")


if __name__ == "__main__":
    print("\n" + "🏥 MEDICAL REPORT EXAMPLES ".center(60, "=") + "\n")
    
    # 1. Brief report (most common - sent via SMS)
    example_brief_report()
    print("\n")
    
    # 2. Short summary (single SMS)
    example_short_summary()
    print("\n")
    
    # 3. Printable report (for clinic/doctor)
    example_printable_report()
    
    print("\n" + "=" * 60)
    print("✓ All examples generated successfully!")
    print("=" * 60)
