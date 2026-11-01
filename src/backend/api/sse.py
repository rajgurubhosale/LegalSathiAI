import json


def sse(data, event: str | None = None) -> str:
    """ STREM RESPOSE IN FORMAT FOR UI"."""

    message = ""
    if event:
        message += f"event: {event}\n"
        
    message += f"data: {json.dumps(data)}\n\n"  
    return message