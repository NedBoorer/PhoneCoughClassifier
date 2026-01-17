"""
Application Startup Tasks
Handles model preloading and other initialization
"""
import logging
from app.config import settings
from app.ml.model_hub import get_model_hub

logger = logging.getLogger(__name__)


def preload_ml_models():
    """
    Preload ML models on startup to eliminate first-call latency.

    Without preloading, the first call takes 30-60 seconds as models load.
    With preloading, all calls are consistently fast (3-8 seconds).
    """
    if not settings.preload_models_on_startup:
        logger.info("Model preloading disabled - models will load on first use")
        return

    try:
        hub = get_model_hub()
        hub.preload_models(
            respiratory=settings.enable_respiratory_screening,
            tuberculosis=settings.enable_tuberculosis_screening,
            parkinsons=settings.enable_parkinsons_screening,
            depression=settings.enable_depression_screening,
        )
        logger.info("✅ ML models preloaded successfully")
    except Exception as e:
        logger.error(f"❌ Model preloading failed: {e}")
        logger.warning("Models will load lazily on first request (slower)")


def run_startup_tasks():
    """Run all startup tasks"""
    logger.info("Running startup tasks...")
    preload_ml_models()
    logger.info("Startup tasks complete")
