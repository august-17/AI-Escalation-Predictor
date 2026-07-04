"""
Main entry point for the AI Escalation Predictor application.
"""
from __future__ import annotations

import logging

from app.application import Application

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    """Application entry point."""
    
    app = Application()
    app.run()


if __name__ == "__main__":
    main()