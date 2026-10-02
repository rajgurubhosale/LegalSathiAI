from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os

load_dotenv()


def get_model(model: str = "openai/gpt-oss-120b", temperature: float = 0):
    
    keyname = "GROQ_API_KEY"
    api_key = os.getenv(keyname)
    
    if not api_key:
        raise ValueError(f"{keyname} not found in environment.")
    
    return ChatGroq(model=model, api_key=api_key, temperature=temperature)

