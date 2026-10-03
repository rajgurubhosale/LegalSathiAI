from pydantic import BaseModel, Field


class SubQueriesSchema(BaseModel):
    questions: list[str] = Field(
        description="Standalone questions covering all parts of the original question",
        min_length=1,
    )

