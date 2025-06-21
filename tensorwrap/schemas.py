from typing import List, Optional, Literal
from pydantic import BaseModel

class ProblemSpec(BaseModel):
    """Specification of a kernel problem to optimize."""
    name: str
    shape_a: List[int]
    shape_b: List[int]
    dtype: str

class KernelCandidate(BaseModel):
    """Represents a candidate kernel implementation."""
    id: Optional[int] = None
    problem: str
    round: int
    code: str
    idea: str
    correct: bool = False
    latency_ms: Optional[float] = None

class Score(BaseModel):
    """Represents the score for a kernel candidate."""
    candidate_id: int
    correct: bool
    latency_ms: float