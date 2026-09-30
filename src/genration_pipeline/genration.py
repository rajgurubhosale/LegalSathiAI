from src.gen.model import get_model
from src.gen.new_prompts import system_msg
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
    


def generate_answer(question: str, context: str, chat_history: list = None) -> dict:
    """Generate an answer using retrieved context + prior conversation history."""
    if chat_history is None:
        chat_history = []
        

    llm = get_model()
    chain = system_msg | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question,
    })

    updated_history = chat_history + [
        HumanMessage(content=question),
        AIMessage(content=answer),
    ]

    return {"answer": answer, "question": question, "chat_history": updated_history}

