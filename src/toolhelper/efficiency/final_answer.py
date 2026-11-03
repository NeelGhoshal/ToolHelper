"""E9: final answer at the turn limit.

When `max_turns` runs out, ask once more, with no tools offered, for an answer from what
was gathered. The baseline lost runs that had the facts but never stopped calling tools
(med-007, x-008 on 2026-11-02). Variants: on/off, and the limit itself (6/8/12) set via
`AgentConfig.max_turns`.

Evidence is the runs that motivated it: baseline runs that ended at `max_turns`.
"""

from __future__ import annotations

from typing import Any

from toolhelper.agent.hooks import BaseStrategy, Task
from toolhelper.context.model import EvidenceRef
from toolhelper.llm.base import Message

PROMPT = ("You have run out of tool calls. Answer the original question now, using only facts and "
          "identifiers from the tool results above. If they do not answer it, say what is missing.")


class FinalAnswerAtLimit(BaseStrategy):
    name = "E9_final_answer"

    def __init__(self, evidence_runs: list[str], prompt: str = PROMPT) -> None:
        if not evidence_runs:
            raise ValueError("E9 needs evidence: run ids of baseline runs that ended at max_turns")
        self.evidence_runs = list(evidence_runs)
        self.prompt = prompt

    def config(self) -> dict[str, Any]:
        return {"prompt": self.prompt, "evidence_runs": len(self.evidence_runs)}

    def at_turn_limit(self, task: Task, messages: list[Message]) -> Message | None:
        self.run.transform("history", before=0, after=self.prompt,
                           evidence=[EvidenceRef(kind="run", ref=r) for r in self.evidence_runs],
                           note="closing prompt at max_turns; tools withdrawn")
        return Message(role="user", content=self.prompt)
