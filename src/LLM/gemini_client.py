"""Gemini client used for household task decomposition."""

import os
import time
from pathlib import Path

from google import genai
from google.genai import errors, types

from .prompt import SYSTEM_PROMPT, build_user_prompt


DEFAULT_MODEL = "gemini-3.7-flash"
MAX_RETRY_ATTEMPTS = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
API_KEY_FILE = Path(__file__).with_name("api_key.txt")


def load_api_key() -> str:
    """Load the key from the environment or the ignored local text file."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key

    if API_KEY_FILE.exists():
        api_key = API_KEY_FILE.read_text(encoding="utf-8").strip()
        if api_key:
            return api_key

    raise RuntimeError(
        "Missing Gemini API key. Set GEMINI_API_KEY or add it to "
        f"{API_KEY_FILE}."
    )


def create_client():
    """Create a Gemini client using the configured local credential."""
    return genai.Client(api_key=load_api_key())


def decompose_instruction(instruction: str, model=DEFAULT_MODEL) -> str:
    """Ask Gemini for a JSON task DAG, falling back on transient failures."""
    client = create_client()
    try:
        last_error = None
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=build_user_prompt(instruction),
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        automatic_function_calling=(
                            types.AutomaticFunctionCallingConfig(disable=True)
                        ),
                    ),
                )
            except errors.APIError as error:
                if error.code not in RETRYABLE_STATUS_CODES:
                    raise RuntimeError(
                        f"Gemini API failed ({error.code}): {error.message}"
                    ) from None
                last_error = error
                if attempt < MAX_RETRY_ATTEMPTS:
                    delay = 2 ** (attempt - 1)
                    print(
                        f"Gemini {model} is temporarily unavailable "
                        f"({error.code}); retrying in {delay}s "
                        f"({attempt}/{MAX_RETRY_ATTEMPTS})..."
                    )
                    time.sleep(delay)
                continue

            if not response.text:
                raise RuntimeError(
                    f"Gemini {model} returned no text."
                )
            return response.text

        raise RuntimeError(
            f"Gemini {model} is temporarily unavailable after "
            f"{MAX_RETRY_ATTEMPTS} attempts. Please run the command again "
            "shortly."
        ) from last_error
    finally:
        client.close()
