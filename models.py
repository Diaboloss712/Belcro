from pydantic import BaseModel
from typing import List, Optional

class CodeLine(BaseModel):
    line: int
    tag: Optional[str]
    classes: List[str]
    available_horizontal: List[str]

class ChatResponse(BaseModel):
    answer: str
    code: str
    lines: List[CodeLine]
    horizontal_options: List[str] = []
    selected_components: List[str] = []

class ChatRequest(BaseModel):
    question: str
    context: Optional[str] = None
