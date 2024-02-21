import time
from typing import Dict, Literal, Optional

from torch.cuda import Event

Unit = Literal["ms", "s"]


MS_TO_UNIT: Dict[Unit, float] = {"ms": 1.0, "s": 1e-3}


class Timer:
    def __init__(self, cuda: bool, unit: Unit) -> None:
        self.is_cuda = cuda

        if self.is_cuda:
            self.start_event: Event = Event(enable_timing=True)
            self.end_event: Event = Event(enable_timing=True)
        else:
            self.start_time: Optional[int] = None

        self._result: Optional[float] = None
        self.unit: Unit = unit

    def start(self):
        if self.is_cuda:
            self.start_event.record()
        else:
            self.start_time = time.monotonic_ns()

    def end(self) -> float:
        if self.is_cuda:
            self.end_event.record()
            result_ms = self.start_event.elapsed_time(self.end_event)
        else:
            end_time = time.monotonic_ns()
            assert self.start_time is not None
            result_ms = (end_time - self.start_time) * 1e-6

        self._result = result_ms * MS_TO_UNIT[self.unit]

        return self._result

    @property
    def elapsed_time(self) -> float:
        assert self._result is not None
        return self._result

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.end()
