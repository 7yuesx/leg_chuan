"""`python -m offline` 入口，等价于 `python compute_fit.py`（新版实现）。"""

from .pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
