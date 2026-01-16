"""
TwiML Utilities - Shared helpers for voice webhooks
Consolidates duplicate code across webhook routers
"""
from fastapi.responses import Response
from twilio.twiml.voice_response import VoiceResponse


def twiml_response(twiml: VoiceResponse) -> Response:
    """
    Convert TwiML VoiceResponse to FastAPI Response.

    This helper is used across all webhook routers to ensure consistent
    response formatting for Twilio.

    Args:
        twiml: Twilio VoiceResponse object

    Returns:
        FastAPI Response with XML content type
    """
    return Response(content=str(twiml), media_type="application/xml")
