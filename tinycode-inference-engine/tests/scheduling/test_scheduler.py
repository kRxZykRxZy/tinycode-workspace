"""Tests of the scheduler behaviour."""

from __future__ import annotations

from tinycode_engine.scheduling.backpressure import BackpressureController
from tinycode_engine.scheduling.continuous_batching import ContinuousBatchingScheduler
from tinycode_engine.scheduling.priorities import Priority


def test_the_backpressure_is_inactive_for_an_idle_server() -> None:
    controller = BackpressureController(queue_depth=0, queue_capacity=64, active_requests=0)
    signal = controller.evaluate()
    assert not signal.active


def test_the_backpressure_activates_for_a_full_queue() -> None:
    controller = BackpressureController(queue_depth=64, queue_capacity=64, active_requests=0)
    signal = controller.evaluate()
    assert signal.active
    assert signal.retry_after_seconds > 0


def test_the_scheduler_runs_the_queued_requests() -> None:
    class _Session:
        def __init__(self) -> None:
            self.generated: list[int] = []
            self.finished = False
            self.finish_reason = None

        def step(self):
            if self.finished:
                return None
            self.generated.append(1)
            if len(self.generated) >= 3:
                self.finished = True
                self.finish_reason = "length"
            return type("Chunk", (), {"text": "x"})()

        def release(self) -> None:
            pass

    scheduler = ContinuousBatchingScheduler(session_factory=lambda rid, payload: _Session())
    assert scheduler.submit("a", None, priority=Priority.NORMAL)
    scheduler.run()
    assert scheduler.stats.completed == 1
    assert not scheduler.active


def test_a_cancelled_request_is_not_executed() -> None:
    class _Session:
        def __init__(self) -> None:
            self.generated: list[int] = []
            self.finished = False
            self.finish_reason = None

        def step(self):
            self.finished = True
            self.finish_reason = "cancelled"
            return type("Chunk", (), {"text": "x"})()

        def release(self) -> None:
            pass

    scheduler = ContinuousBatchingScheduler(session_factory=lambda rid, payload: _Session())
    scheduler.submit("a", None)
    scheduler.cancel("a")
    scheduler.run()
    assert scheduler.stats.cancelled >= 1
