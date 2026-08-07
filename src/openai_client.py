from openai import OpenAI

from src.config import OPENAI_API_KEY

client = OpenAI(
    api_key=OPENAI_API_KEY
)