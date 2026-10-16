from langchain_groq import ChatGroq

from dotenv import load_dotenv
import os

load_dotenv()


def get_model(model: str = "openai/gpt-oss-120b", temperature: float = 0, api_key: str | None = None):
    api_key = api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment and no api_key passed.")

    return ChatGroq(
        model=model,
        api_key=api_key,
        temperature=temperature,
        timeout=60,
        max_retries=1,
    )
