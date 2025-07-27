from pydantic import BaseModel
from typing import List, Optional

class CodeLine(BaseModel):
    line: int
    tag: str
    classes: List[str]
    available_horizontal: List[str]

class ChatResponse(BaseModel):
    answer: str
    code: str
    lines: List[CodeLine]

class ChatRequest(BaseModel):
    question: str
    session_id: str
    context: Optional[str] = None