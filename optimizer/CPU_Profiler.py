import time
from functools import wraps
import numpy as np
import os as _os
import psutil as _psutil

"""
Profiling ideas:
Wall time (latency): time.perf_counter()

Process memory (RSS / resident set): how much RAM the process holds — psutil

Python heap allocations (only Python objects): tracemalloc

Peak memory: highest RSS during a block — resource (Unix) or periodic sampling with psutil

CPU load % and per-core usage: psutil
"""

def _full_name(func):
    # Module + Class + Method, e.g. package.module.Class.method
    return f"{func.__module__}.{func.__qualname__}()"

# def profile(_func=None, *, store=None, print_line=False):
#     """
#     Flexible profiler decorator.

#     Usage:
#       @profile
#       def f(...): ...

#       @profile(store=my_list)
#       def f(...): ...

#       # Preallocated numpy buffer:
#       buf = np.empty(10000, dtype=float)
#       idx = {"i": 0}  # mutable index
#       @profile(store={"buf": buf, "i": idx})
#       def f(...): ...

#     'store' accepted forms:
#       - None: just print
#       - any object with .append -> append (name, seconds)
#       - dict {"buf": np.ndarray, "i": {"i": int}} -> write to buf[i], then i += 1
#         (you can also pass "i": {"i": 0} once and reuse)
#     """
#     def _decorator(func):
#         name = _full_name(func)

#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             t0 = time.process_time()
#             try:
#                 return func(*args, **kwargs)
#             finally:
#                 dt = time.process_time() - t0

#                 # 1) Python list (or anything with append)
#                 if store is not None and hasattr(store, "append"):
#                     store.append(dt)

#                 # 2) Preallocated NumPy buffer
#                 elif isinstance(store, dict) and "buf" in store and "i" in store:
#                     buf = store["buf"]
#                     iref = store["i"]  # expecting a mutable dict like {"i": 0}
#                     i = iref["i"]
#                     if not isinstance(buf, np.ndarray):
#                         raise TypeError("store['buf'] must be a numpy.ndarray")
#                     if i >= buf.size:
#                         # grow (amortized), or raise—your call
#                         new = np.empty(max(2*buf.size, 1), dtype=buf.dtype)
#                         new[:buf.size] = buf
#                         store["buf"] = buf = new
#                     buf[i] = dt
#                     iref["i"] = i + 1

#                 # 3) Nothing stored
#                 if print_line:
#                     print(f"[PROFILE] {name} took {dt:.5f} sec (CPU)")

#         return wrapper

#     # Allow @profile and @profile(...)
#     if _func is not None and callable(_func):
#         return _decorator(_func)
#     return _decorator


def profile(_func=None, *, store=None, print_line=False):
    """
    Flexible profiler decorator.

    Usage:
      @profile
      def f(...): ...

      @profile(store=my_list)
      def f(...): ...

    'store' accepted forms:
      - None: just print
      - any object with .append -> append (name, seconds)
      If 'store' is a dict, we will also record:
        - 'rss'   : Resident Set Size (MB) after the call
        - 'vms'   : Virtual Memory Size (MB) after the call (if psutil available)
    """
    def _decorator(func):
        name = _full_name(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.process_time()
            try:
                return func(*args, **kwargs)
            finally:
                dt = time.process_time() - t0

                # --- memory snapshot (done AFTER call) ---
                # psutil is optional; fall back gracefully if not available
                rss = None
                vms = None
                try:
                    _p = _psutil.Process(_os.getpid())
                    _mi = _p.memory_info()  # rss, vms always present
                    rss = _mi.rss / (1024.0 * 1024.0)
                    vms= _mi.vms / (1024.0 * 1024.0)
                except Exception:
                    pass

                # 1) Python list (or anything with append)
                if store is not None and hasattr(store, "append"):
                    store.append(dt)

                # 2) Preallocated NumPy buffer
                elif isinstance(store, dict):
                    store.setdefault("cpu_time", [])
                    store.setdefault("rss", [])
                    store.setdefault("vms", [])

                    store["cpu_time"].append(dt)
                    store["rss"].append(rss)
                    store["vms"].append(vms)

                # 3) Nothing stored
                if print_line:
                    mem_txt = ""
                    if rss is not None:
                        mem_txt += f" | RSS: {rss:.2f} MB"
                    if vms is not None:
                        mem_txt += f" | VMS: {vms:.2f} MB"
                    print(f"[PROFILE] {name} took {dt:.5f} sec (CPU){mem_txt}")

        return wrapper

    # Allow @profile and @profile(...)
    if _func is not None and callable(_func):
        return _decorator(_func)
    return _decorator