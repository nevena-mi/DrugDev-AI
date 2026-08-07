from openai_client import client
from config import CHAT_MODEL

response = client.responses.create(
    model=CHAT_MODEL,
    input="Say hello in one sentence."
)

print(response.output_text)