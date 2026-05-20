from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import DATABASE_URL
from .logger import logger

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

try:
    with engine.connect() as _conn:
        logger.info("Database connection established")
except Exception as exc:
    logger.error(f"Database connection failed: {exc}")
    raise


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
