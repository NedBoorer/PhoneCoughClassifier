"""
Health Card Generator
Generates visual health report cards using Pillow
"""
import uuid
from pathlib import Path
from typing import Optional

from app.config import settings
from app.utils.image_generator import generate_health_card as create_image

def generate_health_card(result, language: str = "en") -> Optional[str]:
    """
    Generate a health card image for a screening result.
    
    Args:
        result: ComprehensiveHealthResult object
        language: Language code (en, hi, etc.)
        
    Returns:
        Publicly accessible URL path to the generated image, or None if failed.
    """
    try:
        # Generate unique filename
        filename = f"card_{uuid.uuid4().hex}.jpg"
        output_dir = settings.data_dir / "health_cards"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        # Extract details for the card
        # Combine findings from different classifiers
        details = []
        
        from app.utils.i18n import get_text
        
        # Respiratory
        resp = result.screenings.get("respiratory")
        if resp:
            status_key = "status_detected" if resp.detected else "status_normal"
            status = get_text(status_key, language)
            details.append(f"Respiratory: {status}")
        
        # Tuberculosis screening
        tb = result.screenings.get("tuberculosis")
        if tb:
            tb_label = get_text("tb_screening_label", language)
            if tb.detected:
                if tb.severity == "high_risk":
                    tb_status = get_text("tb_high_risk", language)
                elif tb.severity == "moderate_risk":
                    tb_status = get_text("tb_moderate_risk", language)
                else:
                    tb_status = get_text("tb_low_risk", language)
                details.append(f"{tb_label}: ⚠️ {tb_status}")
                # Add DOTS center info for TB detection
                dots_info = get_text("tb_dots_info", language)
                details.append(dots_info)
            else:
                tb_status = get_text("tb_normal", language)
                details.append(f"{tb_label}: ✅ {tb_status}")
            
        # Recommendation
        details.append(result.recommendation)
        
        full_text = "\n".join(details)
        
        # Generate image
        create_image(
            risk_level=result.overall_risk_level,
            details=full_text,
            language=language,
            output_path=output_path,
            optimize_for_2g=True
        )
        
        return str(output_path)
        
    except Exception as e:
        print(f"Failed to generate health card: {e}")
        return None
