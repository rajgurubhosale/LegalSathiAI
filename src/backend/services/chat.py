from langchain_core.messages import AIMessage, HumanMessage


def to_langchain(messages) -> list:
    return [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in messages
    ]
def build_sources(pipeline, docs) -> list[dict]:
    return [
        {"act_name": d.metadata.get("act_name"), "pages": pipeline.document_pages(d)}
        for d in docs
    ]