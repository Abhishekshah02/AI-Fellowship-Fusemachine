import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .logger import logger

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

try:
    with engine.connect() as _conn:
        logger.info("Database connection established successfully")
except Exception as exc:
    logger.error(f"Database connection failed: {exc}")
    raise


def get_db():
    """FastAPI dependency: hand out a DB session, close it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
