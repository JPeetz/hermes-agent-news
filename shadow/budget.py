"""One budget across all attempts, including ambiguous and partial failures."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetLimits:
    max_requests: int = 64
    max_input_tokens: int = 2_000_000
    max_output_tokens: int = 100_000
    deadline_seconds: float = 1800
    max_cost_usd: float | None = None

    def __post_init__(self):
        for key in ("max_requests", "max_input_tokens", "max_output_tokens"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for value in (self.deadline_seconds, self.max_cost_usd):
            if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
                raise ValueError("Budget time/cost limits must be finite and positive")


@dataclass(frozen=True)
class Reservation:
    id: int
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


class RequestBudget:
    def __init__(self, limits: BudgetLimits | None = None, *, clock=time.monotonic):
        self.limits = limits or BudgetLimits()
        self._clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self._attempts: dict[int, dict] = {}

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.limits.deadline_seconds - (self._clock() - self._started))

    def _totals(self):
        return (sum(a["input_tokens"] for a in self._attempts.values()),
                sum(a["output_tokens"] for a in self._attempts.values()),
                sum(a["cost_usd"] or 0.0 for a in self._attempts.values()))

    def reserve(self, *, input_tokens: int, output_tokens: int,
                cost_usd: float | None = None) -> Reservation:
        for value in (input_tokens, output_tokens):
            if type(value) is not int or value < 0:
                raise ValueError("Token reservations must be nonnegative integers")
        if cost_usd is not None and (isinstance(cost_usd, bool) or
                not math.isfinite(cost_usd) or cost_usd < 0):
            raise ValueError("Cost reservation must be finite and nonnegative")
        with self._lock:
            used_in, used_out, used_cost = self._totals()
            if (self.remaining_seconds <= 0 or len(self._attempts) >= self.limits.max_requests or
                    used_in + input_tokens > self.limits.max_input_tokens or
                    used_out + output_tokens > self.limits.max_output_tokens or
                    (self.limits.max_cost_usd is not None and
                     used_cost + (cost_usd or 0) > self.limits.max_cost_usd)):
                raise BudgetExceeded("Experiment request/token/time/cost budget exhausted")
            result = Reservation(len(self._attempts) + 1, input_tokens, output_tokens, cost_usd)
            self._attempts[result.id] = {
                "input_tokens": input_tokens, "output_tokens": output_tokens,
                "cost_usd": cost_usd, "settled": False,
                "input_measured": False, "output_measured": False, "cost_measured": False,
            }
            return result

    def settle(self, reservation: Reservation, *, input_tokens: int | None = None,
               output_tokens: int | None = None, cost_usd: float | None = None) -> None:
        for value in (input_tokens, output_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("Reported usage must be nonnegative integers or unknown")
        if cost_usd is not None and (isinstance(cost_usd, bool) or
                not math.isfinite(cost_usd) or cost_usd < 0):
            raise ValueError("Reported cost must be finite and nonnegative or unknown")
        with self._lock:
            if reservation.id not in self._attempts or self._attempts[reservation.id]["settled"]:
                raise ValueError("Unknown or already-settled reservation")
            row = self._attempts[reservation.id]
            for field, value, measured in (("input_tokens", input_tokens, "input_measured"),
                                           ("output_tokens", output_tokens, "output_measured"),
                                           ("cost_usd", cost_usd, "cost_measured")):
                if value is not None:
                    row[field] = value
                    row[measured] = True
            # An unknown/ambiguous attempt consumes its full conservative reservation.
            row["settled"] = True

    def snapshot(self) -> dict:
        with self._lock:
            rows = list(self._attempts.values())
            used_in, used_out, used_cost = self._totals()
            return {
                "requests": len(rows), "input_tokens_accounted": used_in,
                "output_tokens_accounted": used_out,
                "input_tokens_measured": sum(r["input_tokens"] for r in rows if r["input_measured"]),
                "output_tokens_measured": sum(r["output_tokens"] for r in rows if r["output_measured"]),
                "unknown_usage_attempts": sum(not (r["input_measured"] and r["output_measured"]) for r in rows),
                "cost_usd": used_cost if all(r["cost_measured"] for r in rows) else None,
                "cost_usd_accounted": used_cost,
                "unknown_cost_attempts": sum(not r["cost_measured"] for r in rows),
                "in_flight": sum(not r["settled"] for r in rows),
                "remaining_seconds": self.remaining_seconds,
            }
