from typing import Literal
from pydantic import BaseModel, EmailStr, Field
from uuid import UUID


class UserModel(BaseModel):
    id: int
    username: str
    email: EmailStr
    hashed_password: str
    role: str = "user"
    is_active: bool = True

class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=30)
    email: EmailStr
    password: str = Field(min_length=8, max_length=20)

class UserOut(BaseModel):
    id: int
    username: str
    role: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"



class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1,max_length=4000)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    chat_id: UUID
    chat_history: list[ChatMessage] = Field(default_factory=list, max_length=20)

class Source(BaseModel):
    act_name: str | None = None
    pages: list[int] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    chat_history: list[ChatMessage]
    sources: list[Source]
