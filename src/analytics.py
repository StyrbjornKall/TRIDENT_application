import os
import uuid

import httpx
import streamlit as st
from dotenv import load_dotenv
from loguru import logger

# Loads SUPABASE_URL/SUPABASE_KEY from a .env file.
# Locally this is the .env in the project root (default dotenv search).
# In deployment (SciLifeLab Serve), ENV_FILE points at a .env uploaded to
# the mounted persistent volume, which is never baked into the image.
load_dotenv(dotenv_path=os.environ.get("ENV_FILE"))


def _get_supabase_config() -> tuple[str, str] | None:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        logger.warning("SUPABASE_URL/SUPABASE_KEY not set - page view analytics disabled.")
        return None
    return url.rstrip("/"), key


def _get_session_id() -> str:
    if "analytics_session_id" not in st.session_state:
        st.session_state["analytics_session_id"] = str(uuid.uuid4())
    return st.session_state["analytics_session_id"]


def log_page_view(page_name: str) -> None:
    config = _get_supabase_config()
    if config is None:
        return
    url, key = config
    try:
        response = httpx.post(
            f"{url}/rest/v1/page_views",
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            json={"session_id": _get_session_id(), "page": page_name},
            timeout=5,
        )
        response.raise_for_status()
    except Exception as e:
        logger.warning(f"Failed to log page view to Supabase: {e}")
