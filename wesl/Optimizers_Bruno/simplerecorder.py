# Developed by Bruno de Miranda Boer
# University of Massachusetts Boston, WESLab
# Supervised by Professor Rafael Vallota Rodrigues

import csv
import time
from pathlib import Path


class SimpleRecorder:
    def __init__(self, prob, out_path,
                 x_name='x', y_name='y',
                 aep_name='aep_comp.aep',
                 obj_name='penalty_obj.objective',
                 iter_name='driver_iter.ncg_iter',
                 viol_name='rms_viol'):
        self.prob = prob
        self.out_path = Path(out_path)
        self.x_name = x_name
        self.y_name = y_name
        self.aep_name = aep_name
        self.obj_name = obj_name
        self.iter_name = iter_name
        self.viol_name = viol_name
        self.t0 = None
        self.eval_count = 0
        self._header_written = False

    def start(self):
        self.t0 = time.time()

    def log(self):
        if self.t0 is None:
            raise RuntimeError("SimpleRecorder.start() was not called.")

        t_rel = time.time() - self.t0
        self.eval_count += 1

        x = self.prob.get_val(self.x_name)
        y = self.prob.get_val(self.y_name)
        aep = float(self.prob.get_val(self.aep_name))
        obj = float(self.prob.get_val(self.obj_name))
        it = float(self.prob.get_val(self.iter_name))
        viol = float(self.prob.get_val(self.viol_name))

        write_header = (not self._header_written and not self.out_path.exists())
        with self.out_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["eval", "time_s", "iter", "obj", "aep", "x", "y", 'rms_viol'])
                self._header_written = True
            writer.writerow([
                self.eval_count,
                t_rel,
                it,
                obj,
                aep,
                ";".join(map(str, x)),
                ";".join(map(str, y)),
                viol
            ])

####Class optimized for less memory usage (just aep )
# class SimpleRecorder:
#     def __init__(self, prob, out_path,
#                  aep_name='aep_comp.aep'):
#         self.prob = prob
#         self.out_path = Path(out_path)
#         self.aep_name = aep_name
#         self.t0 = None
#         self.eval_count = 0
#         self._header_written = False

#     def start(self):
#         self.t0 = time.time()

#     def log(self):
#         if self.t0 is None:
#             raise RuntimeError("SimpleRecorder.start() was not called.")

#         t_rel = time.time() - self.t0
#         self.eval_count += 1

#         aep = float(self.prob.get_val(self.aep_name))

#         write_header = (not self._header_written and not self.out_path.exists())
#         with self.out_path.open("a", newline="") as f:
#             writer = csv.writer(f)
#             if write_header:
#                 writer.writerow(["eval", "time_s", "aep"])
#                 self._header_written = True
#             writer.writerow([
#                 self.eval_count,
#                 t_rel,
#                 aep
#             ])

#                  iter_name='driver_iter.opt_iter',
#                  viol_name='rms_viol',
#                  mode='all',
#                  time_decimals=3,
#                  obj_decimals=2,
#                  aep_decimals=6,
#                  viol_decimals=6,
#                  xy_decimals=5):
#         self.prob = prob
#         self.out_path = Path(out_path)

#         self.x_name = x_name
#         self.y_name = y_name
#         self.aep_name = aep_name
#         self.obj_name = obj_name
#         self.iter_name = iter_name
#         self.viol_name = viol_name

#         self.mode = mode

#         self.time_decimals = time_decimals
#         self.obj_decimals = obj_decimals
#         self.aep_decimals = aep_decimals
#         self.viol_decimals = viol_decimals
#         self.xy_decimals = xy_decimals

#         self.t0 = None
#         self.eval_count = 0

#         self._first_row_written = False
#         self._last_row_full = None

#     def start(self):
#         self.t0 = time.time()

#     def _header(self):
#         return ["eval", "time_s", "iter", "obj", "aep", "x", "y", "rms_viol"]

#     def _need_header(self):
#         return (not self.out_path.exists()) or (self.out_path.stat().st_size == 0)

#     def _append_row(self, row):
#         with self.out_path.open("a", newline="") as f:
#             writer = csv.writer(f)
#             if self._need_header():
#                 writer.writerow(self._header())
#             writer.writerow(row)

#     def _fmt(self, value, ndigits):
#         return f"{float(value):.{ndigits}f}"

#     def _fmt_int(self, value):
#         return str(int(round(float(value))))

#     def _serialize_vec(self, v):
#         return ";".join(f"{float(val):.{self.xy_decimals}f}" for val in v)

#     def _make_row(self, include_xy=True):
#         if self.t0 is None:
#             raise RuntimeError("SimpleRecorder.start() was not called.")

#         t_rel = time.time() - self.t0
#         self.eval_count += 1

#         aep = self.prob.get_val(self.aep_name)
#         obj = self.prob.get_val(self.obj_name)
#         it = self.prob.get_val(self.iter_name)
#         viol = self.prob.get_val(self.viol_name)

#         if include_xy:
#             x = self.prob.get_val(self.x_name)
#             y = self.prob.get_val(self.y_name)
#             x_val = self._serialize_vec(x)
#             y_val = self._serialize_vec(y)
#         else:
#             x_val = ''
#             y_val = ''

#         return [
#             self.eval_count,
#             self._fmt(t_rel, self.time_decimals),
#             self._fmt_int(it),
#             self._fmt(obj, self.obj_decimals),
#             self._fmt(aep, self.aep_decimals),
#             x_val,
#             y_val,
#             self._fmt(viol, self.viol_decimals)
#         ]

#     def log(self):
#         if self.mode == 'all':
#             is_first = (self.eval_count == 0)

#             row = self._make_row(include_xy=is_first)
#             self._append_row(row)

#             last_row_full = list(row)
#             if not is_first:
#                 x = self.prob.get_val(self.x_name)
#                 y = self.prob.get_val(self.y_name)
#                 last_row_full[5] = self._serialize_vec(x)
#                 last_row_full[6] = self._serialize_vec(y)

#             self._last_row_full = last_row_full

#         elif self.mode == 'first_last':
#             row = self._make_row(include_xy=True)

#             if not self._first_row_written:
#                 self._append_row(row)
#                 self._first_row_written = True

#             self._last_row_full = row

#         else:
#             raise ValueError(f"Unknown recorder mode: {self.mode}")

#     def close(self):
#         if self._last_row_full is None:
#             return

#         last_eval = int(self._last_row_full[0])
#         if last_eval > 1:
#             self._append_row(self._last_row_full)