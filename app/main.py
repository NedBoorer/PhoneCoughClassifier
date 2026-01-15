"""
Phone Cough Classifier - FastAPI Main Application
Voice agent pipeline for cough classification using real COUGHVID dataset
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle - startup and shutdown"""
    # Startup
    logger.info("🚀 Starting Phone Cough Classifier...")
    logger.info(f"   Environment: {settings.environment}")
    logger.info(f"   Base URL: {settings.base_url}")
    
    # Create directories
    settings.data_dir
    settings.models_dir
    settings.recordings_dir
    
    # Initialize database
    try:
        from app.database.database import init_db
        await init_db()
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.warning(f"Database init skipped: {e}")
    
    # Load ML model
    try:
        from app.ml.classifier import get_classifier
        classifier = get_classifier()
        logger.info(f"✓ ML classifier loaded: {classifier.model_type}")
    except Exception as e:
        logger.warning(f"ML model not loaded: {e}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Phone Cough Classifier...")
    try:
        from app.database.database import close_db
        await close_db()
    except Exception:
        pass


# Create FastAPI app
app = FastAPI(
    title="Phone Cough Classifier",
    description="Voice agent pipeline for cough classification using COUGHVID dataset",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================
# Core Endpoints
# ==================

@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Phone Cough Classifier",
        "version": "1.0.0",
        "description": "Voice agent for cough classification",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health = {
        "status": "healthy",
        "environment": settings.environment,
        "components": {}
    }
    
    # Check database
    try:
        from app.database.database import async_session_maker
        async with async_session_maker() as session:
            await session.execute("SELECT 1")
        health["components"]["database"] = "connected"
    except Exception as e:
        health["components"]["database"] = f"error: {str(e)}"
    
    # Check ML model
    try:
        from app.ml.classifier import get_classifier
        classifier = get_classifier()
        health["components"]["ml_model"] = classifier.model_type
    except Exception as e:
        health["components"]["ml_model"] = f"not loaded: {str(e)}"
    
    # Check Twilio config
    health["components"]["twilio"] = "configured" if settings.twilio_account_sid else "not configured"
    
    # Check OpenAI config
    health["components"]["openai"] = "configured" if settings.openai_api_key else "not configured"
    
    return health


# ==================
# Include Routers
# ==================

# Twilio webhooks
try:
    from app.api.twilio_webhooks import router as twilio_router
    app.include_router(twilio_router, prefix="/twilio", tags=["Twilio Webhooks"])
    logger.info("✓ Twilio webhooks loaded")
except ImportError as e:
    logger.warning(f"Twilio webhooks not loaded: {e}")

# India accessibility webhooks
try:
    from app.api.india_webhooks import router as india_router
    app.include_router(india_router, prefix="/india", tags=["India Accessibility"])
    logger.info("✓ India webhooks loaded")
except ImportError as e:
    logger.warning(f"India webhooks not loaded: {e}")

# Test endpoints
try:
    from app.api.test_endpoints import router as test_router
    app.include_router(test_router, prefix="/test", tags=["Testing"])
    logger.info("✓ Test endpoints loaded")
except ImportError as e:
    logger.warning(f"Test endpoints not loaded: {e}")


# ==================
# Static Files
# ==================
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
