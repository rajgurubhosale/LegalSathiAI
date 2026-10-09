from pathlib import Path

import streamlit as st

_BROWSER_SESSION = st.components.v2.component(
    "browser_session",
    js=Path(__file__).with_name("browser_session.js").read_text(encoding="utf-8"),
)


def sync_browser_session(token=None, *, clear=False):
    """Keep the signed token in this browser tab across page refreshes."""
    previous = st.session_state.get("browser_auth", {}).get("result")
    result = _BROWSER_SESSION(
        key="browser_auth",
        data={
            "action": "clear" if clear else "save" if token else "read",
            "token": token,
            "previous": previous,
        },
        default={"result": None},
        on_result_change=lambda: None,
        height=0,
    )
    return result.result
