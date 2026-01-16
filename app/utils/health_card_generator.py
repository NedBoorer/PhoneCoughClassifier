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
            # TODO: Get translated text
            status = "Detected" if resp.detected else "Normal"
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
