from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
from pydantic import BaseModel, Field


@dataclass
class Message:
    source: str
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    priority: float = 1.0
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)


class GlobalState(BaseModel):
    arousal_level: float = Field(default=0.5, ge=0.0, le=1.0)
    mood_valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    current_goals: List[Dict] = Field(default_factory=list)
    active_context: Dict[str, Any] = Field(default_factory=dict)
    safety_status: str = "normal"

    class Config:
        arbitrary_types_allowed = True


class HormonalState:
    """
    Lightweight cross-module summary of hidden-layer state.

    NOTE (design boundary, see spec "Core Emergence Principle"): this class
    is a convenience container for passing a *summary* between modules. It
    must never be handed to prometheus.py's decision logic as a substitute
    for a synthesized felt state -- only synthesizer.py's output (composite
    arousal/valence/dominance -> named felt state) is allowed to influence
    orchestration decisions. This class existing is not, by itself, a
    license to read core.py's raw values into agent logic.
    """

    def __init__(self):
        self.arousal_level: float = 0.5
        self.stress_level: float = 0.3
        self.mood_valence: float = 0.0
