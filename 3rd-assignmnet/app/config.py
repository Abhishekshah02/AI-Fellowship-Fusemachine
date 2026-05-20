import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

AGENT_MAX_RETRIES = int(os.getenv("AGENT_MAX_RETRIES", "3"))
QUERY_ROW_LIMIT = int(os.getenv("QUERY_ROW_LIMIT", "200"))
