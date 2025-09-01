from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)
speech_file_path = Path(__file__).parent / "speech.mp3"

with client.audio.speech.with_streaming_response.create(
    model="gpt-4o-mini-tts",
    voice="nova",
    input="Hello, thank you for contacting our support team. We understand your concern and we’re here to help. Could you please provide a few more details so we can quickly resolve the issue for you?",
    instructions="Speak calmly, slowly and professionally in french, like a customer support representative",
) as response:
    response.stream_to_file(speech_file_path)