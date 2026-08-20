"""Check that the watchdog tells a working process from a hung one.

The timer fires on wall clock but only cancels the run when the process has
burned less than a twentieth of a core since the last check. A busy child and a
sleeping child should land on opposite sides of that line.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rspipe.runner import _cpu_seconds  # noqa: E402

WINDOW = 4.0
BUSY = 0.05 * WINDOW

CASES = {
    "busy": "import time\nt=time.time()\nx=0\nwhile time.time()-t<8: x+=1",
    "idle": "import time; time.sleep(8)",
}


def main() -> int:
    bad = 0
    for label, code in CASES.items():
        p = subprocess.Popen([sys.executable, "-c", code])
        a = _cpu_seconds(p) or 0.0
        time.sleep(WINDOW)
        b = _cpu_seconds(p) or 0.0
        keeps = (b - a) > BUSY
        want = label == "busy"
        ok = keeps == want
        bad += not ok
        print(f"{label}: grew {b - a:6.3f} cpu-s over {WINDOW} s "
              f"(threshold {BUSY:.2f}) -> "
              f"{'keep running' if keeps else 'cancel'}   "
              f"{'ok' if ok else 'WRONG'}")
        p.kill()
        p.wait()
    print("dead process reads as", _cpu_seconds(p))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
