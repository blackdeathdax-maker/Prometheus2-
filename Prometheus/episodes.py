"""
episodes.py — Package F: minimal episodic trace + consolidation replay.

Not full autobiographical memory. A ring of structured act outcomes:
  (pulse, focus/lemma, op, body_improved, info_score, causal_made)

Replay on consolidation: if the same (lemma, op) pattern repeats with
consistent improvement, mild re-credit to op L and optional link boost.
No graph spam: episode *nodes* only if repeat_count >= NODE_MIN (off by default).
"""
from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple


class EpisodeLog:
    CAP = 256
    REPLAY_MIN_REPEATS = 3
    REPLAY_MAX_PATTERNS = 12
    # Optional graph nodes (disabled by default — enable via flag)
    MINT_NODES = False
    NODE_MIN_REPEATS = 8

    def __init__(self, cap: int = CAP):
        self.buf: Deque[dict] = deque(maxlen=max(32, int(cap)))
        self.pattern_counts: Dict[str, int] = {}
        self.pattern_improve: Dict[str, int] = {}  # net improved count
        self.last_replay: Dict[str, Any] = {}

    def _key(self, lemma: str, op: str) -> str:
        return f"{str(lemma or '').casefold().strip()}|{str(op or '').upper()}"

    def record(
        self,
        pulse: int,
        focus_id: str = "",
        lemma: str = "",
        op: str = "",
        body_improved: Optional[bool] = None,
        info_score: float = 0.0,
        causal_made: int = 0,
        note: str = "",
    ) -> None:
        lemma = str(lemma or focus_id or "").strip()
        op_u = str(op or "HOLD").upper()
        row = {
            "pulse": int(pulse),
            "focus_id": str(focus_id or ""),
            "lemma": lemma,
            "op": op_u,
            "body_improved": body_improved,
            "info_score": float(info_score or 0.0),
            "causal_made": int(causal_made or 0),
            "note": str(note or "")[:80],
        }
        self.buf.append(row)
        k = self._key(lemma, op_u)
        self.pattern_counts[k] = int(self.pattern_counts.get(k, 0)) + 1
        if body_improved is True or float(info_score or 0) > 0.15:
            self.pattern_improve[k] = int(self.pattern_improve.get(k, 0)) + 1
        elif body_improved is False or float(info_score or 0) < -0.15:
            self.pattern_improve[k] = int(self.pattern_improve.get(k, 0)) - 1

    def recent(self, n: int = 12) -> List[dict]:
        return list(self.buf)[-n:]

    def top_patterns(self, n: int = 8) -> List[Tuple[str, int, int]]:
        """Return (key, count, net_improve) sorted by count."""
        rows = [
            (k, int(c), int(self.pattern_improve.get(k, 0)))
            for k, c in self.pattern_counts.items()
        ]
        rows.sort(key=lambda t: (-t[1], -t[2]))
        return rows[:n]

    def replay(
        self,
        credit_fn=None,
        link_boost_fn=None,
    ) -> dict:
        """Consolidation replay: re-credit consistent successful (lemma, op) patterns.

        credit_fn(op, improved, context, magnitude) — e.g. operators.credit_evidence
        link_boost_fn(lemma) — optional mild residual/edge warm
        """
        report = {
            "patterns_seen": 0,
            "replayed": 0,
            "ops": [],
            "skipped_weak": 0,
        }
        ranked = self.top_patterns(self.REPLAY_MAX_PATTERNS * 2)
        for key, count, net in ranked:
            if report["replayed"] >= self.REPLAY_MAX_PATTERNS:
                break
            report["patterns_seen"] += 1
            if count < self.REPLAY_MIN_REPEATS:
                report["skipped_weak"] += 1
                continue
            # Need net positive improvement signal
            if net < 2:
                report["skipped_weak"] += 1
                continue
            parts = key.split("|", 1)
            lemma = parts[0] if parts else ""
            op = parts[1] if len(parts) > 1 else "HOLD"
            mag = min(1.2, 0.25 + 0.08 * min(count, 10) + 0.05 * min(net, 8))
            if credit_fn is not None:
                try:
                    credit_fn(op, True, context=lemma[:64], magnitude=mag)
                except Exception:
                    pass
            if link_boost_fn is not None and lemma:
                try:
                    link_boost_fn(lemma)
                except Exception:
                    pass
            report["replayed"] += 1
            report["ops"].append({"key": key, "count": count, "net": net, "mag": round(mag, 3)})

        # Soft decay pattern tables so old habits don't dominate forever
        for k in list(self.pattern_counts.keys()):
            self.pattern_counts[k] = max(0, int(self.pattern_counts[k] * 0.92))
            if self.pattern_counts[k] == 0:
                self.pattern_counts.pop(k, None)
                self.pattern_improve.pop(k, None)
            else:
                self.pattern_improve[k] = int(
                    self.pattern_improve.get(k, 0) * 0.92
                )

        self.last_replay = report
        return report

    def report(self) -> dict:
        return {
            "buffer": len(self.buf),
            "recent": self.recent(8),
            "top_patterns": [
                {"key": k, "count": c, "net_improve": n}
                for k, c, n in self.top_patterns(8)
            ],
            "last_replay": dict(self.last_replay or {}),
        }
