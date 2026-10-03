import logging
import sys
from pathlib import Path
from uuid import uuid4

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="LegalSaathi", page_icon="⚖️", layout="wide")


@st.cache_resource(show_spinner=False)
def load_pipeline():
    # Load the models once, when the first question is submitted.
    from src.genration_pipeline.pipeline import LegalSaathiPipeline
    from src.utils.main_utils import read_config_file

    settings = read_config_file()["evaluation"]
    return LegalSaathiPipeline(
        top_n=settings["top_n"],
        rerank_k=settings["rerank_k"],
        min_score=settings.get("min_score"),
    )


def new_chat():
    chat_id = uuid4().hex
    st.session_state.chats[chat_id] = {
        "title": "New chat", "messages": [], "history": [],
    }
    st.session_state.current_chat = chat_id


def switch_chat(chat_id):
    st.session_state.current_chat = chat_id


def show_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View sources", icon=":material/library_books:"):
                for source in message["sources"]:
                    pages = ", ".join(map(str, source["pages"]))
                    st.write(f"📄 {source['pdf']} — PDF pages: {pages}")


if "chats" not in st.session_state:
    st.session_state.chats = {}
    new_chat()

pdfs = sorted((ROOT / "PDF_DATA").glob("*.pdf"))
pdf_names = {pdf.stem: pdf.name for pdf in pdfs}

with st.sidebar:
    st.title("LegalSaathi", icon="⚖️")
    st.button("New chat", icon=":material/add:", on_click=new_chat, width="stretch")
    st.subheader("Chat history")
    for chat_id, saved_chat in reversed(list(st.session_state.chats.items())):
        st.button(
            saved_chat["title"],
            key=chat_id,
            type="primary" if chat_id == st.session_state.current_chat else "secondary",
            on_click=switch_chat,
            args=(chat_id,),
            width="stretch",
        )
    st.caption("Chats are kept for this browser session.")
    st.divider()
    st.subheader("Document vault")
    with st.container(height=280, border=False):
        for pdf in pdfs:
            st.write(f"📄 {pdf.name}")
        if not pdfs:
            st.caption("No PDFs found in PDF_DATA.")

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
    if not chat["messages"]:
        chat["title"] = question[:50]
    message = {"role": "user", "content": question}
    chat["messages"].append(message)
    show_message(message)

    try:
        with st.spinner("Searching documents… The first question also loads the models."):
            pipeline = load_pipeline()
            docs = pipeline.retrieve_context(question, return_docs=True)
            context = "\n\n".join(doc.page_content for doc in docs)
            result = pipeline.generate_answer(question, context, chat["history"])

        # Combine duplicate PDF references while keeping their real PDF pages.
        sources = {}
        for doc in docs:
            act = doc.metadata["act_name"]
            name = pdf_names.get(act, act)
            pages = doc.metadata.get("pages") or [doc.metadata["page"]]
            sources.setdefault(name, set()).update(int(page) for page in pages)

        chat["messages"].append({
            "role": "assistant",
            "content": result["answer"],
            "sources": [{"pdf": name, "pages": sorted(pages)} for name, pages in sources.items()],
        })
        chat["history"] = result["chat_history"]
    except Exception:
        logging.exception("LegalSaathi could not answer the question")
        chat["messages"].append({
            "role": "assistant",
            "content": "I couldn't complete this request. Please check the terminal error and try again.",
        })
    st.rerun()
