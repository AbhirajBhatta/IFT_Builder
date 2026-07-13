from pathlib import Path

from sqlmodel import SQLModel, create_engine, Session
from app.config import get_settings

settings = get_settings()

# Ensure the SQLite file's parent directory exists before the engine tries
# to open it — data/pdfs/ and data/output/ are created by their own owning
# modules (routes.py, formatter.py), so db/ needs the same treatment here.
Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{settings.db_path}",
    echo=False,
    connect_args={"check_same_thread": False},
)


def create_db_and_tables():
    """Call once at startup to initialise all tables."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """FastAPI dependency — yields a DB session per request."""
    with Session(engine) as session:
        yield session
