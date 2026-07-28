import asyncio
import sys
import os
from pathlib import Path

# Add the project root to sys.path to allow importing from backend
# Path(__file__).resolve() is /path/to/backend/scripts/init_db.py
# .parent is /path/to/backend/scripts
# .parent.parent is /path/to/backend
# .parent.parent.parent is /path/to
# Wait, if I'm in /home/shefo90/Desktop/pathira/pathira-app/backend/scripts/init_db.py
# I want /home/shefo90/Desktop/pathira/pathira-app in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from backend.db.base import Base
from backend.db.session import engine
# Import all models to register them with Base.metadata
from backend.db.models.user import User
from backend.db.models.alert import Alert
from backend.db.models.companion import Companion
from backend.db.models.emergency import EmergencyEvent
from backend.db.models.face import Face
from backend.db.models.location import Location
from backend.db.models.refresh_token import RefreshToken
from backend.db.models.device_token import DeviceToken
from backend.db.models.notification_request import NotificationRequest
from backend.db.models.vision_detection import VisionDetection
from backend.db.models.support_session import SupportSession

async def init_db():
    try:
        async with engine.begin() as conn:
            print("Creating tables...")
            await conn.run_sync(Base.metadata.create_all)
            print("Tables created successfully.")
    except Exception as e:
        print(f"Error creating tables: {e}")
    finally:
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(init_db())
