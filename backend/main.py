from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi import _rate_limit_exceeded_handler

from backend.core.config import settings
from backend.api.endpoints import (
    auth, users, companions, alerts, tracking, faces, emergency,
    notifications, vision, support, websockets, voice,
)
from backend.api.limiter import limiter

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Serve ML model weights as static files so the phone can download them
# on first launch (e.g. GET /static/models/yolov8n.onnx).
_weights_dir = Path(settings.ML_WEIGHTS_DIR)
if _weights_dir.exists():
    app.mount("/static/models", StaticFiles(directory=str(_weights_dir)), name="ml_models")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/users", tags=["User Management"])
app.include_router(companions.router, prefix="/api/companions", tags=["Companions (main)"])
app.include_router(companions.companion_router, prefix="/api/companion", tags=["Companions (companion)"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(alerts.companion_alerts_router, prefix="/api/companion", tags=["Alerts (companion)"])
app.include_router(tracking.router, prefix="/api/tracking", tags=["Location Tracking"])
app.include_router(faces.router, prefix="/api/companion", tags=["Faces"])
app.include_router(emergency.router, prefix="/api/emergency", tags=["Emergency"])
app.include_router(notifications.router, prefix="/api/notifications", tags=["Push Notifications"])
app.include_router(vision.router, prefix="/api/vision", tags=["Vision AI"])
app.include_router(support.router, prefix="/api/support", tags=["Support"])
app.include_router(voice.router, prefix="/api/voice", tags=["Voice Commands"])
# WebSocket routes already include the /api/... prefix in their paths.
app.include_router(websockets.router, tags=["WebSockets"])


@app.get("/", tags=["Health Check"])
async def root():
    return {"message": "Welcome to Pathira API", "status": "running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
