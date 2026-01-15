from PIL import Image, ImageDraw, ImageFont
import textwrap
from pathlib import Path
from typing import Optional

def generate_health_card(
    risk_level: str,
    details: str,
    language: str = "en",
    output_path: Path = Path("health_card.jpg"),
    optimize_for_2g: bool = True
) -> Path:
    """
    Generates a visual health report card optimized for rural 2G/3G networks.

    OPTIMIZATION FOR RURAL AREAS:
    - Reduced size: 400x500 (vs 800x1000) = 1/4 file size
    - JPEG format with 60% quality (vs PNG) = 10x smaller
    - Target: <50KB (loads in ~30 seconds on 2G)
    - Still readable on feature phones

    Args:
        risk_level: 'low', 'medium', 'high'
        details: Health recommendation text
        language: Language code (en, hi, ta, etc.)
        output_path: Where to save the image
        optimize_for_2g: Use low-bandwidth optimizations

    Returns:
        Path to generated image file
    """

    # Color Scheme (Traffic light system for low literacy)
    colors = {
        "low": ("#4CAF50", "#E8F5E9"),     # Green - Safe
        "mild": ("#4CAF50", "#E8F5E9"),    # Green - Safe
        "normal": ("#4CAF50", "#E8F5E9"),  # Green - Safe
        "medium": ("#FFC107", "#FFF8E1"),  # Amber - Caution
        "moderate": ("#FFC107", "#FFF8E1"), # Amber - Caution
        "high": ("#F44336", "#FFEBEE"),    # Red - Urgent
        "severe": ("#F44336", "#FFEBEE"),  # Red - Urgent
        "urgent": ("#D32F2F", "#FFCDD2")   # Dark Red - Emergency
    }

    base_color, bg_color = colors.get(risk_level.lower(), ("#9E9E9E", "#F5F5F5"))

    # Canvas - OPTIMIZED for 2G/3G networks
    if optimize_for_2g:
        width, height = 400, 500  # Quarter the pixels = 1/4 file size
    else:
        width, height = 800, 1000  # Original high-res

    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Scale factors for optimized layout
    scale = 0.5 if optimize_for_2g else 1.0

    # Header Background
    header_height = int(100 * (1 / scale))
    draw.rectangle([(0, 0), (width, header_height)], fill=base_color)

    # Text Setup - Scale font sizes for smaller images
    try:
        font_header = ImageFont.truetype("arial.ttf", int(30 / scale))
        font_body = ImageFont.truetype("arial.ttf", int(20 / scale))
        font_small = ImageFont.truetype("arial.ttf", int(15 / scale))
    except IOError:
        # Fallback to default font
        font_header = ImageFont.load_default()
        font_body = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Title
    title_text = "Health Report"
    if language == "hi":
        title_text = "स्वास्थ्य रिपोर्ट"
    elif language == "ta":
        title_text = "சுகாதார அறிக்கை"

    draw.text((int(25 / scale), int(35 / scale)), title_text, fill="white", font=font_header)

    # Large Risk Indicator (Traffic light circle for low literacy)
    circle_y = int(125 / scale)
    circle_radius = int(60 / scale)
    draw.ellipse(
        [(int((width / 2) - circle_radius), circle_y),
         (int((width / 2) + circle_radius), circle_y + circle_radius * 2)],
        fill=base_color,
        outline="black",
        width=2
    )

    # Risk Level Text
    risk_display = risk_level.upper()
    risk_emoji = {"low": "✓", "mild": "✓", "normal": "✓",
                  "moderate": "!", "medium": "!",
                  "high": "!!", "severe": "!!", "urgent": "!!!"}
    emoji = risk_emoji.get(risk_level.lower(), "")

    draw.text(
        (int(width / 2) - int(30 / scale), circle_y + int(20 / scale)),
        f"{emoji}",
        fill="white",
        font=font_header
    )

    # Recommendations Box
    rec_start_y = int(260 / scale)
    rec_text = details[:150] if optimize_for_2g else details  # Truncate for smaller images
    wrapped_text = textwrap.fill(rec_text, width=int(40 / scale))

    draw.text((int(25 / scale), rec_start_y), wrapped_text, fill="black", font=font_body)

    # Footer
    footer_y = int(450 / scale) if optimize_for_2g else int(900 / scale)
    footer = "AI Screening - See Doctor"
    draw.text((int(25 / scale), footer_y), footer, fill="gray", font=font_small)

    # Save with optimization
    if optimize_for_2g:
        # JPEG with 60% quality for 2G/3G networks
        # Target: <50KB (loads in ~30 seconds on 2G)
        img.save(output_path, format='JPEG', quality=60, optimize=True)
    else:
        # Original PNG for high-bandwidth
        img.save(output_path, format='PNG')

    return output_path
