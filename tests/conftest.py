"""Shared pytest fixtures/setup.

Loads variables from the project-root ``.env`` file (e.g. ``OPENAI_API_KEY``,
``LOGFIRE_TOKEN``) into ``os.environ`` before the test session starts, so
tests can rely on them without requiring them to be exported manually in the
shell.

Existing environment variables always take precedence over ``.env`` values
(``override=False``), so CI secrets or manually exported values are never
clobbered.
"""

from pathlib import Path

from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(dotenv_path=_ENV_PATH, override=False)
