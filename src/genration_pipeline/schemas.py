from pydantic import BaseModel, Field
from typing import Literal



class SubQueriesSchema(BaseModel):
    questions: list[str] = Field(
        description="Standalone questions covering all parts of the original question",
        min_length=1,
    )

