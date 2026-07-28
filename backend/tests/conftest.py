import asyncio
import inspect
import uuid
from typing import AsyncIterator

# Allow the reserved `.test` TLD for test fixtures (pytest+<uuid>@example.test).
# email-validator >= 2.1 rejects special-use TLDs by default; opt into test mode
# at module load time so EmailStr validation accepts these addresses.
import email_validator
email_validator.TEST_ENVIRONMENT = True

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.limiter import limiter
from backend.db.session import AsyncSessionLocal, engine
# Import every model so SQLAlchemy can resolve all string-referenced relationships
# before the autouse cleanup fixture issues a delete() against User.
from backend.db.models.user import User
from backend.db.models.alert import Alert  # noqa: F401
from backend.db.models.companion import Companion  # noqa: F401
from backend.db.models.emergency import EmergencyEvent  # noqa: F401
from backend.db.models.face import Face  # noqa: F401
from backend.db.models.location import Location  # noqa: F401
from backend.db.models.refresh_token import RefreshToken  # noqa: F401
from backend.main import app


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    # Tests share a single in-process limiter; reset between tests so rate
    # limit state from a prior test cannot bleed into the next one.
    limiter.reset()
    yield


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def db() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session


@pytest.fixture
def unique_email() -> str:
    return f"pytest+{uuid.uuid4().hex}@example.test"


def _sync_cleanup_test_users() -> None:
    """Synchronous cleanup used for non-async tests (e.g. WebSocket tests
    using Starlette's sync TestClient). Runs in a separate thread with its
    own event loop so it can safely dispose the engine without colliding
    with the calling thread's running loop.
    """
    import threading

    error: list[BaseException] = []

    def _runner() -> None:
        async def _do() -> None:
            await engine.dispose()
            async with AsyncSessionLocal() as session:
                await session.execute(
                    delete(User).where(User.email.like("pytest+%@example.test"))
                )
                await session.commit()
            await engine.dispose()

        try:
            asyncio.run(_do())
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    t = threading.Thread(target=_runner)
    t.start()
    t.join()
    if error:
        raise error[0]


@pytest_asyncio.fixture(autouse=True)
async def _cleanup_test_users(request):
    is_async_test = inspect.iscoroutinefunction(request.node.function)
    yield
    if not is_async_test:
        # Sync tests (e.g. WebSocket tests using Starlette's TestClient)
        # pin connections to a TestClient worker-thread loop; awaiting
        # engine.dispose() from pytest-asyncio's loop would deadlock or
        # raise "another operation is in progress". Hand off to a thread
        # with its own loop instead.
        _sync_cleanup_test_users()
        return

    async with AsyncSessionLocal() as session:
        await session.execute(
            delete(User).where(User.email.like("pytest+%@example.test"))
        )
        await session.commit()
    # Dispose pooled asyncpg connections so the next test's event loop gets
    # fresh connections (otherwise pooled conns are pinned to the prior loop
    # and SQLAlchemy raises "attached to a different loop").
    await engine.dispose()
