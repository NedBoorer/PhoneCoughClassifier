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
        
        # Respiratory
        resp = result.screenings.get("respiratory")
        if resp:
            from app.utils.i18n import get_text
            status_key = "status_detected" if resp.detected else "status_normal"
            status = get_text(status_key, language)
            
            # Label for Respiratory
            # We don't have a specific label key for "Respiratory" in the provided i18n, 
            # assuming "Respiratory: {status}" format, but let's try to localize the label if possible.
            # For now, we will keep "Respiratory" english distinct or add a key if needed. 
            # Looking at i18n.py, we don't have "Respiratory" key.
            # Let's just localize the status part effectively.
            details.append(f"Respiratory: {status}")
            
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
