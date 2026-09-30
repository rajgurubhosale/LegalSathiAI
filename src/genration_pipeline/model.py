from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os

load_dotenv()


def get_model(model: str = "gemini-3.1-flash-lite", temperature: float = 0):
    
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment.")
    
    return ChatGoogleGenerativeAI(model=model, api_key=api_key, temperature=temperature)

