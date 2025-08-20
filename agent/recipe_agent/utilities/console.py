import sys, io, threading
from contextlib import contextmanager
from typing import Any, Callable, List, Tuple

class LiveBar:
    """A simple class to render a live, updating progress bar in the console."""
    def __init__(self, prefix: str = "", width: int = 28):
        self.prefix = prefix
        self.width = width
        self.pct = 0.0
        self._last_render_len = 0
        self._render()

    def tick_towards(self, target_pct: float, step: float = 0.7) -> None:
        """Gradually advances the progress bar towards a target percentage."""
        target_pct = max(0.0, min(100.0, float(target_pct)))
        if self.pct < target_pct:
            self.pct = min(target_pct, self.pct + step)
            self._render()

    def _render(self) -> None:
        """Internal method to draw the progress bar to the console."""
        pct_int = int(self.pct)
        filled = int((pct_int / 100.0) * self.width)
        bar = "█" * filled + "-" * (self.width - filled)
        line = f"{self.prefix} [{bar}] {pct_int:3d}%"
        sys.stdout.write("\r" + line + " " * max(0, self._last_render_len - len(line)))
        sys.stdout.flush()
        self._last_render_len = len(line)

    def finish(self) -> None:
        """Completes the progress bar and moves to a new line."""
        self.pct = 100.0
        self._render()
        sys.stdout.write("\n")
        sys.stdout.flush()


class Progress:
    """A thread-safe class for tracking the progress of a task."""
    def __init__(self, total_units: int):
        self.total = max(1, int(total_units))
        self.done = 0
        self.lock = threading.Lock()

    def advance(self, n: int = 1) -> None:
        """Increments the number of completed units in a thread-safe manner."""
        with self.lock:
            self.done = min(self.total, self.done + int(n))

    def fraction(self) -> float:
        """Returns the current progress as a fraction (0.0 to 1.0)."""
        with self.lock:
            return 0.0 if self.total == 0 else min(1.0, self.done / self.total)


@contextmanager
def suppress_stdout_stderr():
    """A context manager that temporarily redirects stdout and stderr to a buffer."""
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
        yield
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err


def run_in_thread(fn: Callable, *args, **kwargs) -> Tuple[Any, List[Any], List[BaseException]]:
    """
    Runs a function in a separate daemon thread and returns a tuple
    containing the thread object, a list for the function's result, and a
    list for any errors that occurred.
    """
    result_holder: List[Any] = []
    error_holder: List[BaseException] = []

    def _target():
        try:
            result_holder.append(fn(*args, **kwargs))
        except BaseException as e:
            error_holder.append(e)

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    return th, result_holder, error_holder