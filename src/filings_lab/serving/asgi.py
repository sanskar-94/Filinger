"""ASGI entry point for uvicorn or gunicorn: ``uvicorn filings_lab.serving.asgi:app``."""
from .app import create_app

app = create_app()
