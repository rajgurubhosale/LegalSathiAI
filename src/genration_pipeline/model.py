from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

MESH_KEY = os.getenv("MESH_API_KEY")
MESH_BASE_URL = "https://api.meshapi.ai/v1"


def get_model_meshapi(
    model_name: str = "google/gemini-2.5-flash-lite",
    temperature: float = 0.0,
    api_key: str | None = None,
):
    key = api_key or MESH_KEY
    return ChatOpenAI(
        model=model_name,
        api_key=key,
        base_url=MESH_BASE_URL,
        temperature=temperature,
        streaming=True,
    )


def get_model(temperature: float = 0, api_key: str | None = None):
    return get_model_meshapi(temperature=temperature, api_key=api_key)


summarizer = get_model_meshapi(model_name="google/gemini-2.5-flash-lite", temperature=0.0)

chit_chat_model = get_model_meshapi(model_name="google/gemini-2.5-flash-lite", temperature=0.2)

from typesafe_sdk import Choice, TypeSafeClient
amos_client = TypeSafeClient(
    base_url="https://api.amos.scira.ai",
    api_key="amos",  
    model="amos-1-medium",

)
