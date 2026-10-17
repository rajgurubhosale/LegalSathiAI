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
    list_user_vault_documents, load_messages, login, read_document,
    read_user_vault_document, register, upload_personal_documents,
    delete_user_vault_document, clear_user_vault,
)
from frontend.session import sync_browser_session

st.set_page_config(page_title="LegalSaathi", page_icon="⚖️", layout="wide")
st.html(Path(__file__).with_name("styles.css"))


def sign_out():
    st.session_state.clear_browser_auth = True
    for key in ("access_token", "username", "chats", "recent_chats", "chat_error", "current_chat", "active_view", "vault_page", "vault_current_page", "selected_pdf", "selected_user_doc", "show_sources", "settings_show_sources"):
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
        "title": "New chat", "messages": [], "is_draft": True,
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
        "title": title, "messages": messages, "is_draft": False,
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


def open_personal_vault():
    st.session_state.active_view = "personal_vault"
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
        st.button(
            "Personal vault", icon=":material/lock:", on_click=open_personal_vault,
            type="primary" if st.session_state.get("active_view") == "personal_vault" else "secondary",
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

                    with column, st.container(key=f"vault_card_{pdf}", border=True, height=320, gap="small"):
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

if st.session_state.get("active_view") == "personal_vault":
    if selected_user_doc := st.session_state.get("selected_user_doc"):
        st.button("Back to personal vault", icon=":material/arrow_back:", on_click=lambda: st.session_state.pop("selected_user_doc", None))
        st.subheader(selected_user_doc["display_name"], icon=":material/description:")
        try:
            content = read_user_vault_document(selected_user_doc["id"], st.session_state.access_token)
        except requests.HTTPError as error:
            if error.response is not None and error.response.status_code == 401:
                session_expired()
            st.error("Could not load this document. Return to personal vault and try again.")
        except requests.RequestException:
            st.error("The backend is unavailable. Please try again shortly.")
        else:
            if selected_user_doc["file_type"] == "pdf":
                st.pdf(content, height=700, key="personal_pdf_reader")
            else:
                st.download_button(
                    label=f"Download {selected_user_doc['display_name']}",
                    data=content,
                    file_name=selected_user_doc["display_name"],
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    icon=":material/download:",
                    type="primary",
                )
                st.info("Document loaded! Click the button above to download the DOCX file.")
        st.stop()

    st.title("Personal vault", icon=":material/lock:")
    st.caption("Upload, manage, and search your private legal documents.")

    if upload_notice := st.session_state.pop("vault_upload_success", None):
        st.success(upload_notice)

    # 1. Fetch user documents from database
    try:
        user_docs = list_user_vault_documents(st.session_state.access_token)
    except requests.HTTPError as error:
        if error.response is not None and error.response.status_code == 401:
            session_expired()
        st.error("Could not load your vault documents. Please refresh shortly.")
        user_docs = []
    except requests.RequestException:
        st.error("Backend is unavailable. Please refresh shortly.")
        user_docs = []

    MAX_FILES = 5
    MAX_FILE_SIZE_MB = 20
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

    # 2. Compact upload section
    with st.expander("📤 Upload New Documents (PDF, DOCX)", expanded=(len(user_docs) == 0)):
        st.caption("Supported formats: **PDF, DOCX** &nbsp;•&nbsp; Limit: **Up to 5 files** &nbsp;•&nbsp; **Max 20 MB** per file")
        uploader_key = st.session_state.get("uploader_key", 0)
        uploaded_files = st.file_uploader(
            "Choose documents to upload",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key=f"personal_vault_uploader_{uploader_key}",
            label_visibility="collapsed",
            help="Select up to 5 documents in PDF or DOCX format (maximum 20 MB each).",
        )

        if uploaded_files:
            file_count = len(uploaded_files)
            count_exceeded = file_count > MAX_FILES
            oversized_files = [f for f in uploaded_files if f.size > MAX_FILE_SIZE_BYTES]

            if count_exceeded:
                st.error(
                    f"⚠️ **Too many files:** You selected {file_count} files. "
                    f"A maximum of {MAX_FILES} files can be uploaded at a time. Please remove the extra files."
                )

            if oversized_files:
                for f in oversized_files:
                    size_mb = f.size / (1024 * 1024)
                    st.error(
                        f"⚠️ **File too large:** `{f.name}` is **{size_mb:.1f} MB** "
                        f"(exceeds the {MAX_FILE_SIZE_MB} MB limit)."
                    )

            if not count_exceeded and not oversized_files:
                st.success(f"✓ {file_count} document{'s' if file_count > 1 else ''} validated and ready to upload.")

                for f in uploaded_files:
                    icon = "📄" if f.name.lower().endswith(".pdf") else "📝"
                    size_kb = f.size / 1024
                    size_label = f"{size_kb / 1024:.2f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"
                    st.html(
                        f'<div style="display:flex; align-items:center; justify-content:space-between; '
                        f'padding:8px 12px; margin:4px 0; background:#f4f9f6; border:1px solid #d8ebd0; border-radius:8px;">'
                        f'<span>{icon} <strong>{escape(f.name)}</strong></span>'
                        f'<span style="color:#527364; font-size:0.85rem;">{size_label}</span>'
                        f'</div>'
                    )

                st.caption("ℹ️ **Please stay on this screen while indexing.** Leaving this tab will stop processing any remaining documents.")
                st.space("small")
                if st.button(
                    f"Upload {file_count} document{'s' if file_count > 1 else ''} to Vault",
                    key="btn_personal_vault_upload",
                    type="primary",
                    icon=":material/upload:",
                ):
                    with st.status(f"Processing {file_count} document(s)... Please do not leave this page.", expanded=True) as status_box:
                        try:
                            st.write("Uploading and embedding into your vector vault...")
                            res = upload_personal_documents(uploaded_files, st.session_state.access_token)
                            status_box.update(label="✓ Complete! Documents are ready.", state="complete", expanded=False)
                            st.session_state["uploader_key"] = uploader_key + 1
                            st.session_state["vault_upload_success"] = res.get("message", "Uploaded successfully!")
                            st.rerun()
                        except Exception as e:
                            status_box.update(label="❌ Interrupted or Failed", state="error", expanded=False)
                            st.error(f"Upload failed: {e}")

    # 3. Documents Library (User Vault)
    st.space("medium")
    if not user_docs:
        with st.container(border=True):
            st.html("""
                <div style="text-align:center; padding: 24px 12px;">
                    <div style="font-size: 2.2rem; margin-bottom: 8px;">📂</div>
                    <strong style="font-size: 1.1rem; color: #233831;">Your Vault is Empty</strong>
                    <p style="color: #527364; font-size: 0.9rem; margin-top: 6px;">
                        Upload private legal documents, contracts, or notices above to search them in your chats.
                    </p>
                </div>
            """)
    else:
        ch1, ch2 = st.columns([0.72, 0.28], vertical_alignment="center")
        with ch1:
            st.subheader(f"Your Documents ({len(user_docs)})", icon=":material/folder_shared:")
        with ch2:
            with st.popover("Clear all vault data", icon=":material/delete_sweep:"):
                st.caption("⚠️ **Are you sure?** Permanently delete all documents from database, disk, and vector vault.")
                b_yes, b_no = st.columns(2)
                with b_yes:
                    if st.button("Yes, Delete All", type="primary", key="btn_confirm_clear_vault", width="stretch"):
                        try:
                            res = clear_user_vault(st.session_state.access_token)
                            st.session_state["vault_upload_success"] = res.get("message", "Personal vault cleared successfully.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to clear vault: {e}")
                with b_no:
                    if st.button("No, Cancel", key="btn_cancel_clear_vault", width="stretch"):
                        st.rerun()

        for doc in user_docs:
            doc_id = doc["id"]
            name = doc["display_name"]
            ftype = doc["file_type"].lower()
            icon = "📄" if ftype == "pdf" else "📝"
            size_kb = (doc.get("file_size") or 0) / 1024
            size_label = f"{size_kb / 1024:.2f} MB" if size_kb >= 1024 else f"{size_kb:.1f} KB"

            raw_time = doc.get("uploaded_at")
            if raw_time:
                try:
                    if isinstance(raw_time, str):
                        dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                    else:
                        dt = raw_time
                    time_str = dt.astimezone(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %Y, %I:%M %p IST")
                except Exception:
                    time_str = str(raw_time)
            else:
                time_str = "Recently added"

            with st.container(border=True, key=f"user_doc_card_{doc_id}"):
                c1, c2, c3 = st.columns([0.62, 0.24, 0.14], vertical_alignment="center")
                with c1:
                    st.html(f"""
                        <div style="display:flex; flex-direction:column; gap:4px;">
                            <div style="font-size:1rem; font-weight:600; color:#182822; display:flex; align-items:center; gap:8px;">
                                <span>{icon}</span> <span>{escape(name)}</span>
                            </div>
                            <div style="font-size:0.82rem; color:#527364; display:flex; gap:16px;">
                                <span>📦 {size_label}</span>
                                <span>🕒 {time_str}</span>
                                <span style="background:#eaf4ed; color:#23613e; padding:1px 6px; border-radius:4px; font-weight:500;">✓ Indexed</span>
                            </div>
                        </div>
                    """)
                with c2:
                    btn_label = "Read PDF" if ftype == "pdf" else "Download"
                    btn_icon = ":material/menu_book:" if ftype == "pdf" else ":material/download:"
                    if st.button(btn_label, key=f"view_doc_{doc_id}", icon=btn_icon, width="stretch"):
                        st.session_state.selected_user_doc = doc
                        st.rerun()
                with c3:
                    with st.popover("Delete", icon=":material/delete:"):
                        st.caption(f"Delete **{escape(name)}**? This cannot be undone.")
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button("Yes, Delete", key=f"confirm_del_{doc_id}", type="primary", width="stretch"):
                                try:
                                    delete_user_vault_document(doc_id, st.session_state.access_token)
                                    st.session_state["vault_upload_success"] = f"✓ Deleted '{name}' from your vault."
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Failed to delete: {e}")
                        with col_no:
                            if st.button("No, Cancel", key=f"cancel_del_{doc_id}", width="stretch"):
                                st.rerun()

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
