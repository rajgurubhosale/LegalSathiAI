import json
import re
import sys
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

import streamlit as st
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frontend.api import (
    ask_question, create_chat_draft, get_profile, list_chats, list_documents,
    load_messages, login, read_document, recent_history, register,
)
from frontend.session import sync_browser_session

st.set_page_config(page_title="LegalSaathi", page_icon="⚖️", layout="wide")
st.html(Path(__file__).with_name("styles.css"))


def sign_out():
    st.session_state.clear_browser_auth = True
    for key in ("access_token", "username", "chats", "recent_chats", "chat_error", "current_chat", "active_view", "vault_page", "vault_current_page", "selected_pdf", "show_sources", "settings_show_sources"):
        st.session_state.pop(key, None)


def session_expired():
    sign_out()
    st.session_state.auth_notice = "Your session expired. Please sign in again."
    st.rerun()


clearing_auth = st.session_state.get("clear_browser_auth", False)
browser_auth = sync_browser_session(st.session_state.get("access_token"), clear=clearing_auth)
if clearing_auth and browser_auth and browser_auth["action"] == "clear":
    st.session_state.pop("clear_browser_auth", None)
if not st.session_state.get("access_token") and not clearing_auth:
    if browser_auth is None:
        with st.spinner("Opening your workspace…"):
            st.stop()
    if saved_token := browser_auth.get("token"):
        try:
            profile = get_profile(saved_token)
        except requests.HTTPError as error:
            if error.response is not None and error.response.status_code in (401, 403):
                session_expired()
            st.error("Could not restore your session. Please try again shortly.")
            if st.button("Retry"):
                st.rerun()
            st.stop()
        except requests.RequestException:
            st.error("The backend is unavailable. Please try again shortly.")
            if st.button("Retry"):
                st.rerun()
            st.stop()
        else:
            st.session_state.access_token = saved_token
            st.session_state.username = profile["username"]
            st.rerun()
if browser_auth and not browser_auth.get("available", True):
    st.warning("Your browser is blocking session storage. Allow it to stay signed in on refresh.")

if not st.session_state.get("access_token"):
    with st.container(key="auth_shell"):
        welcome, account = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    with welcome, st.container(key="welcome_panel"):
        st.html("""
            <p class="welcome-kicker">A LITTLE CLARITY. A BETTER NEXT STEP.</p>
            <h1 class="welcome-title">Welcome to<br><span>LegalSaathi.</span></h1>
            <p class="welcome-copy">Understand the law. Know your next step.<br>
            Explore Indian law in everyday language, with answers grounded in your legal documents.</p>
            <div class="welcome-features">
                <div class="welcome-feature"><span class="feature-number">01</span><div>
                    <strong>Answers with context</strong><p>Find the source pages behind each answer.</p>
                </div></div>
                <div class="welcome-feature"><span class="feature-number">02</span><div>
                    <strong>A conversation, not a search</strong><p>Ask a question. Follow up. Get a little more clarity.</p>
                </div></div>
                <div class="welcome-feature"><span class="feature-number">03</span><div>
                    <strong>Your document vault</strong><p>Browse your Indian Acts in one simple workspace.</p>
                </div></div>
            </div>
            <p class="welcome-note">Made for the questions that matter to you.</p>
        """)

    with account, st.container(key="auth_card"):
        st.html("""<div class="auth-brand"><span class="brand-symbol" aria-hidden="true">⚖</span>
            <span class="brand-name">Legal Saathi <span class="brand-ai">AI</span></span></div>""")
        action = st.segmented_control(
            "Account", ["Sign in", "Create account"], default="Sign in",
            label_visibility="collapsed", width="stretch",
        ) or "Sign in"
        creating_account = action == "Create account"
        st.subheader("Create your account" if creating_account else "Sign in to your account")
        st.caption("A little clarity starts here." if creating_account else "Welcome back. Your legal workspace awaits.")
        if notice := st.session_state.pop("auth_notice", None):
            st.info(notice)
        with st.form("authentication", border=False):
            username = st.text_input("Username", placeholder="Your name", max_chars=30, help="3–30 characters") if creating_account else ""
            email = st.text_input("Email", placeholder="you@example.com", icon=":material/mail:")
            password = st.text_input(
                "Password", placeholder="Create a password" if creating_account else "Enter your password",
                type="password", icon=":material/lock:",
                help="8–20 characters" if creating_account else None,
            )
            submitted = st.form_submit_button(action, type="primary", width="stretch", icon=":material/arrow_forward:")
        st.caption("Already have an account? Choose Sign in above." if creating_account else "New here? Choose Create account above to get started.")
        if submitted:
            if not email.strip() or not password:
                st.error("Enter your email and password.")
            elif creating_account and len(username.strip()) < 3:
                st.error("Enter a username with at least 3 characters.")
            else:
                try:
                    if creating_account:
                        register(username.strip(), email.strip(), password)
                    token = login(email.strip(), password)
                    profile = get_profile(token)
                except requests.HTTPError as error:
                    code = error.response.status_code if error.response is not None else None
                    messages = {
                        400: "Incorrect email or password.",
                        401: "Incorrect email or password.",
                        403: "This account is disabled.",
                        409: "This username or email is already registered.",
                        422: "Check your email, username (3–30 characters) and password (8–20 characters).",
                        500: "The sign-in service encountered a problem. Please try again shortly.",
                        503: "The sign-in service is starting. Please try again shortly.",
                    }
                    st.error(messages.get(code, "Could not sign in. Please try again."))
                except requests.RequestException:
                    st.error("The backend is unavailable. Please try again shortly.")
                else:
                    st.session_state.access_token = token
                    st.session_state.username = profile["username"]
                    st.rerun()
    st.stop()


def chat_request_failed(error, message):
    if isinstance(error, requests.HTTPError) and error.response is not None:
        if error.response.status_code == 401:
            session_expired()
    st.session_state.chat_error = message


def refresh_recent_chats():
    try:
        saved = list_chats(st.session_state.access_token)
    except requests.RequestException as error:
        chat_request_failed(error, "Could not refresh recent chats. Please try again.")
        return
    st.session_state.recent_chats = saved
    if st.session_state.get("chat_error") == "Could not refresh recent chats. Please try again.":
        st.session_state.pop("chat_error", None)
    for item in saved:
        if item["chat_id"] in st.session_state.chats:
            chat = st.session_state.chats[item["chat_id"]]
            chat["is_draft"] = False
            chat["title"] = item["title"]


def new_chat():
    try:
        chat_id = create_chat_draft(st.session_state.access_token)
    except requests.RequestException as error:
        chat_request_failed(error, "Could not start a new chat. Please try again.")
        return
    for old_id, chat in list(st.session_state.chats.items()):
        if chat.get("is_draft") and not chat["messages"]:
            del st.session_state.chats[old_id]
    st.session_state.chats[chat_id] = {
        "title": "New chat", "messages": [], "history": [], "is_draft": True,
    }
    st.session_state.current_chat = chat_id
    st.session_state.active_view = "chat"
    st.session_state.pop("chat_error", None)


def switch_chat(chat_id, title):
    try:
        messages = load_messages(st.session_state.access_token, chat_id)
    except requests.RequestException as error:
        chat_request_failed(error, "Could not load this chat. Please try again.")
        return
    st.session_state.chats[chat_id] = {
        "title": title, "messages": messages, "history": [], "is_draft": False,
    }
    st.session_state.current_chat = chat_id
    st.session_state.active_view = "chat"
    st.session_state.pop("chat_error", None)


@st.fragment(run_every="30s", key="recent_chats")
def show_recent_chats():
    if not st.session_state.get("access_token"):
        return
    refresh_recent_chats()
    st.caption("YOUR CONVERSATIONS")
    if chat_error := st.session_state.get("chat_error"):
        st.error(chat_error)
    with st.container(key="chat_history"):
        for item in st.session_state.get("recent_chats", []):
            chat_id = item["chat_id"]
            if st.button(
                item["title"], icon=":material/chat_bubble_outline:",
                key=chat_id,
                type="primary" if (
                    st.session_state.get("active_view", "chat") == "chat"
                    and chat_id == st.session_state.current_chat
                ) else "secondary",
                width="stretch",
            ):
                switch_chat(chat_id, item["title"])
                st.rerun()


def open_vault():
    st.session_state.active_view = "vault"
    st.session_state.pop("selected_pdf", None)


def open_pdf(filename):
    st.session_state.selected_pdf = filename


@st.cache_data
def load_pdf_summaries() -> dict:
    summary_file = ROOT / "src" / "backend" / "services" / "pdf_data_summary.json"
    if summary_file.is_file():
        try:
            with open(summary_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_source_preference():
    st.session_state.show_sources = st.session_state.settings_show_sources


@st.dialog("Settings", on_dismiss="rerun")
def show_settings():
    st.toggle(
        "Show sources", value=st.session_state.get("show_sources", True),
        key="settings_show_sources", on_change=save_source_preference,
    )
    if st.button("Done", type="primary", icon=":material/check:"):
        st.rerun()


def show_message(message):
    avatar = ":material/person:" if message["role"] == "user" else ":material/balance:"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message.get("sources") and st.session_state.get("show_sources", True):
            with st.expander("View sources", icon=":material/library_books:"):
                for source in message["sources"]:
                    name = source.get("act_name") or source.get("pdf") or "Document"
                    pages = source.get("pages", [])
                    if pages:
                        pages_str = ", ".join(map(str, pages))
                        st.write(f"📄 {name} — PDF pages: {pages_str}")
                    else:
                        st.write(f"📄 {name}")


if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    new_chat()
if "current_chat" not in st.session_state:
    st.error(st.session_state.get("chat_error") or "Could not start a new chat.")
    if st.button("Retry"):
        st.rerun()
    st.stop()

with st.sidebar:
    st.html("""<div class="sidebar-brand"><span class="brand-symbol" aria-hidden="true">⚖</span><div>
        <strong>Legal Saathi <span class="brand-ai">AI</span></strong>
        <small>Your legal workspace</small></div></div>""")
    with st.container(key="sidebar_nav"):
        st.button("New chat", icon=":material/add:", on_click=new_chat, width="stretch", type="primary")
        st.button(
            "Document vault", icon=":material/folder:", on_click=open_vault,
            type="primary" if st.session_state.get("active_view") == "vault" else "secondary",
            width="stretch",
        )
    st.space("small")
    show_recent_chats()
    st.html("""<div class="sidebar-note"><strong>Start with a simple question.</strong>
        <p>Explore an Act, understand a right, or find your next step.</p></div>""")

username = (st.session_state.get("username") or "").strip() or "User"
name_parts = username.split()
initials = name_parts[0][0] + (name_parts[-1][0] if len(name_parts) > 1 else "")


@st.fragment(run_every="30s")
def show_header_clock():
    now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    st.html(
        f'<time class="header-clock" datetime="{now.isoformat()}">'
        f'<span class="header-date">{now:%a, %d %b %Y}</span>'
        f'<span class="header-time">{now:%I:%M %p} IST</span></time>'
    )


with st.container(key="account_header", horizontal=True, horizontal_alignment="right", vertical_alignment="center", gap="medium"):
    with st.container(key="account_clock", width="content"):
        show_header_clock()
    with st.container(key="account_control", horizontal=True, width="content", vertical_alignment="center", gap="small"):
        with st.container(key="account_avatar", width="content"):
            with st.popover(initials.upper()):
                if st.button("Settings", key="account_settings", icon=":material/settings:", type="tertiary", width="stretch"):
                    show_settings()
                st.button("Sign out", key="account_sign_out", icon=":material/logout:", on_click=sign_out, type="tertiary", width="stretch")
        with st.container(key="account_identity", width="content"):
            st.html(f'<div class="account-details"><strong class="account-name">{escape(username)}</strong>'
                    '<span class="account-role">User</span></div>')

if st.session_state.get("active_view") == "vault":
    if selected_pdf := st.session_state.get("selected_pdf"):
        st.button("Back to document vault", icon=":material/arrow_back:", on_click=open_vault)
        st.subheader(selected_pdf, icon=":material/description:")
        try:
            pdf_data = read_document(selected_pdf, st.session_state.access_token)
        except requests.HTTPError as error:
            if error.response is not None and error.response.status_code == 401:
                session_expired()
            st.error("Could not open this PDF. Return to the vault and try again.")
        except requests.RequestException:
            st.error("The backend is unavailable. Please try again shortly.")
        else:
            st.pdf(pdf_data, height=700, key="document_reader", alt=f"Read {selected_pdf}")
        st.stop()
    st.title("Document vault", icon=":material/folder:")
    try:
        pdfs = list_documents(st.session_state.access_token)
    except requests.HTTPError as error:
        if error.response is not None and error.response.status_code == 401:
            session_expired()
        st.error("Could not load documents. Please refresh shortly.")
    except requests.RequestException:
        st.error("Backend is starting or unavailable. Please refresh shortly.")
    else:
        if not pdfs:
            st.info("No documents available yet.")
        else:
            st.caption(f"{len(pdfs)} documents · Choose a PDF to read it here.")
            page_count = (len(pdfs) + 8) // 9
            page = 1
            if page_count > 1:
                page = st.selectbox("Page", range(1, page_count + 1), index=min(st.session_state.get("vault_current_page", 1), page_count) - 1, key="vault_page", width=140)
            st.session_state.vault_current_page = page
            summaries = load_pdf_summaries()
            visible_pdfs = pdfs[(page - 1) * 9:page * 9]
            for start in range(0, len(visible_pdfs), 3):
                columns = st.columns(3)
                for column, pdf in zip(columns, visible_pdfs[start:start + 3]):
                    doc = summaries.get(pdf, {})
                    title = doc.get("title", pdf)
                    category = doc.get("category", "LEGAL STATUTE")
                    summary_text = doc.get("summary", "Official statutory legal document.")
                    status = doc.get("status", "System Indexed")
                    tier = doc.get("tier", "Foundational")
                    cat_slug = f"cat-{re.sub(r'[^a-z0-9]+', '-', category.lower()).strip('-')}"

                    with column, st.container(key=f"vault_card_{pdf}", border=True, height=295, gap="small"):
                        st.html(f"""
                            <div class="vault-card-top">
                                <span class="vault-badge {cat_slug}">{escape(category)}</span>
                                <span class="vault-status"><span class="status-dot"></span>{escape(status)}</span>
                            </div>
                            <div class="vault-title" title="{escape(title)}">{escape(title)}</div>
                            <div class="vault-summary" title="{escape(summary_text)}">{escape(summary_text)}</div>
                            <div class="vault-footer">
                                <span class="vault-file-pill" title="{escape(pdf)}">📄 {escape(pdf)}</span>
                                <span class="vault-tier">{escape(tier)}</span>
                            </div>
                        """)
                        st.button("Read PDF", key=f"read_{pdf}", icon=":material/menu_book:", on_click=open_pdf, args=(pdf,), width="stretch")
    st.stop()

chat = st.session_state.chats[st.session_state.current_chat]
question = None

if not chat["messages"]:
    st.title("Ask me anything about Indian law", icon="⚖️", text_alignment="center")
    with st.container(horizontal_alignment="center"):
        st.write("Explore the Indian Acts in your document vault.")
    examples = [
        "What is criminal intimidation under BNS?",
        "How long does an RTI reply take?",
        "What remedies does RERA provide for delayed possession?",
    ]
    with st.container(horizontal=True, horizontal_alignment="center"):
        for example in examples:
            if st.button(example):
                question = example

for message in chat["messages"]:
    show_message(message)

st.caption("General legal information only. Not a substitute for a qualified advocate.")
question = st.chat_input("Ask a legal question…", submit_mode="disable") or question
if question:
    question = question.strip()

if question:
    if not chat["messages"]:
        chat["title"] = question[:50]
    message = {"role": "user", "content": question}
    chat["messages"].append(message)
    show_message(message)

    try:
        with st.chat_message("assistant", avatar=":material/balance:"):
            answer_placeholder = st.empty()
            with st.spinner("Searching documents…"):
                result = ask_question(
                    question, st.session_state.current_chat,
                    st.session_state.access_token,
                    on_token=answer_placeholder.markdown,
                )

        reply = {
            "role": "assistant",
            "content": result.get("answer", ""),
            "sources": result.get("sources", []),
        }
        chat["history"] = recent_history(chat["history"] + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": result.get("answer", "")},
        ])
    except requests.Timeout:
        reply = {"role" : "assistant", "content": "The request timed out. Please try again shortly."}
    except requests.ConnectionError:
        reply = {"role": "assistant", "content": "The backend is starting or unavailable. Please try again shortly."}
    except requests.HTTPError as error:
        detail = "Could not complete the request. Please try again."
        if error.response is not None:
            if error.response.status_code == 401:
                session_expired()
            elif error.response.status_code in (502, 503):
                try:
                    detail = error.response.json().get("detail", detail)
                except ValueError:
                    pass
            elif error.response.status_code == 422:
                detail = "Please enter a question with between 1 and 2,000 characters."
        reply = {"role": "assistant", "content": detail}
    except requests.RequestException:
        reply = {"role": "assistant", "content": "Could not complete the request. Please try again."}
    chat["messages"].append(reply)
    refresh_recent_chats()
    st.rerun()
