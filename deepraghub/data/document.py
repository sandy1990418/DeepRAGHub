# deepraghub/data/document.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class Document(BaseModel):
    content: str = Field(..., description="The content of the document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata associated with the document"
    )
    embedding: Optional[list] = Field(
        default=None, description="The embedding of the document"
    )
