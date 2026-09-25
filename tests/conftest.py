"""Test configuration.

app.config builds its settings singleton at import time, so every value the
tests rely on must be in the environment before app modules are imported.
Setting them here, at conftest import, guarantees that ordering.

These are dummy values — the suite never calls a real API.
"""

import os

from dotenv import load_dotenv

# Load .env before the defaults below. setdefault never overwrites, so real
# credentials win where they exist and dummies fill the gaps. Without this,
# the dummies would clobber .env and the integration tests could never
# authenticate. In CI there is no .env, so everything falls through to the
# dummy values — or to the real secrets the eval workflow injects as env vars.
load_dotenv()

os.environ.setdefault("LLM_PROVIDER", "groq")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("META_PHONE_NUMBER_ID", "1234567890")
os.environ.setdefault("META_ACCESS_TOKEN", "test-meta-token")
os.environ.setdefault("META_VERIFY_TOKEN", "test-verify-token")
os.environ.setdefault("META_APP_SECRET", "test-app-secret")
os.environ.setdefault("TWILIO_ACCOUNT_SID", "AC" + "0" * 32)
os.environ.setdefault("TWILIO_AUTH_TOKEN", "0" * 32)
os.environ.setdefault("CHECKPOINT_DB", "")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

VERIFY_TOKEN = os.environ["META_VERIFY_TOKEN"]
APP_SECRET = os.environ["META_APP_SECRET"]


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def meta_payload() -> dict:
    """A realistically shaped inbound text message from the Cloud API."""

    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "WABA_ID",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"phone_number_id": "1234567890"},
                            "contacts": [
                                {"profile": {"name": "Sio"}, "wa_id": "919902245562"}
                            ],
                            "messages": [
                                {
                                    "from": "919902245562",
                                    "id": "wamid.TEST",
                                    "timestamp": "1700000000",
                                    "type": "text",
                                    "text": {"body": "latest AI news"},
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }
