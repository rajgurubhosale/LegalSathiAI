import json


def sse(data, event: str | None = None) -> str:
    """ STREM RESPOSE IN FORMAT FOR UI"""
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(data)}\n\n"