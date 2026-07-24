from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

def get_model(model: str = "llama-3.3-70b-versatile", temperature: float = 0):
    
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment.")
    
    return ChatGroq(model=model, api_key=api_key, temperature=temperature)

