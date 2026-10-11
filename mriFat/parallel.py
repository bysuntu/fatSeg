"""
Run independent per-slice work on several CPU cores.

The segmentation methods spend most of their time on steps that treat each slice on its
own; map_slices() spreads those over worker processes (threads would not help: much of the
work is plain Python loops, which run one at a time). Steps that compare neighbouring
slices stay in the calling process. The results are the same as running slice by slice.

The pool is started on first use and kept for later runs. On Windows each worker starts
by importing the script that was run (e.g. readSpace_threeButton.py), so scripts that
segment must keep their work under `if __name__ == '__main__':`.

WORKERS = 1 turns it off (everything runs in the calling process).
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

WORKERS = max(1, min((os.cpu_count() or 1) - 1, 8))   # leave one core for the GUI

_pool = None


def _get_pool():
    global _pool
    if _pool is None:
        # One math-library thread per worker: the workers already use the cores, and each
        # extra OpenBLAS thread reserves its own buffers (~500 MB per worker instead of ~50 MB)
        for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ.setdefault(var, '1')
        _pool = ProcessPoolExecutor(max_workers=WORKERS)
    return _pool


def _noop():
    time.sleep(0.5)   # keep this worker busy so each warm-up task starts a new one


def warm_up():
    """Start the worker processes now (they take a few seconds), so the first run doesn't wait.

    On Windows the pool only starts a worker when a task finds none idle, so one task per worker.
    """
    if WORKERS > 1:
        pool = _get_pool()
        for _ in range(WORKERS):
            pool.submit(_noop)


def map_slices(func, *iterables):
    """list(map(func, *iterables)) with the calls spread over WORKERS processes.

    func must be a module-level function; its arguments and result are copied between
    processes. If the pool breaks (e.g. a worker was killed), the work is redone here.
    """
    args = list(zip(*iterables))
    if WORKERS <= 1 or len(args) < 2:
        return [func(*a) for a in args]
    chunk = max(1, len(args) // (4 * WORKERS))
    try:
        return list(_get_pool().map(func, *zip(*args), chunksize=chunk))
    except BrokenProcessPool:
        shutdown()
        return [func(*a) for a in args]


def shutdown():
    """Stop the worker processes (pending work is cancelled)."""
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
