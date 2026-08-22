import os
import httpx
from dotenv import load_dotenv

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# 1. Connection Pooling: Instantiate globally to keep the TCP/SSL connection alive
client = httpx.AsyncClient()

# 2. Async execution
async def transcribe_bytes_async(audio_bytes: bytes, filename: str = "audio.wav") -> dict:
    """Transcribe audio bytes non-blocking."""
    try:
        headers = {"xi-api-key": ELEVENLABS_API_KEY}
        files = {"file": (filename, audio_bytes, "audio/wav")}
        data = {"model_id": "scribe_v1"}

        response = await client.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers=headers,
            files=files,
            data=data
        )

        if response.status_code == 200:
            result = response.json()
            return {"success": True, "transcript": result.get("text", "").strip()}
        else:
            return {"success": False, "error": f"{response.status_code} — {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}