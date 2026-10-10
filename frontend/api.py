import json
import os
from urllib.parse import quote

import requests

MAX_HISTORY_MESSAGES = 6
MAX_HISTORY_CHARS = 12_000

BACKEND_URL = os.getenv("LEGALSAATHI_API_URL", "http://127.0.0.1:8000").rstrip("/")


def recent_history(history):
    """Keep complete recent turns within both the message and text limits."""
    selected = []
    total = 0
    for index in range(len(history) - 2, -1, -2):
        pair = history[index:index + 2]
        size = sum(len(message["content"]) for message in pair)
        if len(selected) + 2 > MAX_HISTORY_MESSAGES or total + size > MAX_HISTORY_CHARS:
            break
        selected = pair + selected
        total += size
    return selected


def register(username, email, password):
    response = requests.post(
        f"{BACKEND_URL}/auth/register",
        json={"username": username, "email": email, "password": password},
        timeout=(5, 10),
    )
    response.raise_for_status()
    return response.json()


def login(email, password):
    response = requests.post(
        f"{BACKEND_URL}/auth/login",
        data={"username": email, "password": password},
        timeout=(5, 10),
    )
    response.raise_for_status()
    return response.json()["access_token"]


def get_profile(token):
    response = requests.get(
        f"{BACKEND_URL}/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=(5, 10),
    )
    response.raise_for_status()
    return response.json()


def create_chat_draft(token):
    response = requests.post(
        f"{BACKEND_URL}/chats/draft",
        headers={"Authorization": f"Bearer {token}"},
        timeout=(5, 10),
    )
    response.raise_for_status()
    return response.json()["chat_id"]


def list_chats(token):
    response = requests.get(
        f"{BACKEND_URL}/chats",
        headers={"Authorization": f"Bearer {token}"},
        timeout=(5, 10),
    )
    response.raise_for_status()
    return response.json()


def load_messages(token, chat_id):
    response = requests.get(
        f"{BACKEND_URL}/chats/{quote(str(chat_id), safe='')}/messages",
        headers={"Authorization": f"Bearer {token}"},
        timeout=(5, 10),
    )
    response.raise_for_status()
    return response.json()


def ask_question(question, chat_id, token, on_token=None):
    with requests.post(
        f"{BACKEND_URL}/chat/stream",
        headers={"Authorization": f"Bearer {token}"},
        json={"question": question, "chat_id": chat_id},
        timeout=(5, 180),
        stream=True,
    ) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        answer = []
        sources = []
        event = "message"
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line:
                event = "message"
            elif line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                payload = json.loads(line[5:].strip())
                if event == "error":
                    raise requests.HTTPError(str(payload), response=response)
                if event == "sources":
                    sources = payload
                elif event == "done":
                    return {"answer": "".join(answer), "sources": sources}
                elif event == "message":
                    answer.append(payload)
                    if on_token is not None:
                        on_token("".join(answer))
        raise requests.RequestException("Chat stream ended before completion.", response=response)


def list_documents(token):
    response = requests.get(
        f"{BACKEND_URL}/documents",
        headers={"Authorization": f"Bearer {token}"},
        timeout=(5, 10),
    )
    response.raise_for_status()
    return response.json()


def read_document(filename, token):
    response = requests.get(
        f"{BACKEND_URL}/documents/{quote(filename, safe='')}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=(5, 30),
    )
    response.raise_for_status()
    return response.content

