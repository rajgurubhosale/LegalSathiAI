from pydantic import BaseModel, Field
from typing import Literal



class SubQueriesSchema(BaseModel):
    questions: list[str] = Field(
        description="Standalone questions covering all parts of the original question",
        min_length=1,
    )


class RouteDecision(BaseModel):
    reason: str = Field(description="One short sentence explaining the decision. Write this first.")
    destination: Literal["guardrail", "chit_chat", "rag_retrieval"] = Field(
        description=(  
            "guardrail = user wants help committing or hiding a crime; "
            "chit_chat = greeting, thanks, or a question about this chat itself; "
            "rag_retrieval = any legal question or question about uploaded files. "
            "If unsure between chit_chat and rag_retrieval, choose rag_retrieval."
        )
    )