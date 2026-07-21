# Developed by Bruno de Miranda Boer
# University of Massachusetts Boston, WESLab
# Supervised by Professor Rafael Vallota Rodrigues

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FRAME_NAME_RE = re.compile(r'cycle(\d+)_iter(\d+)(?:_(\w+))?\.png$')


def _sorted_iteration_frames(plots_dir):
    """
    Collects every per-iteration plot in plots_dir (as saved by FarmPlotter during the
    SGD run) in chronological order, separating out the MILP warm-start final frame
    (tagged 'milp_final' by Warm_Start_Cables.py) so it can be appended as its own
    closing segment instead of being sorted into the middle of the sequence.
    """
    iteration_frames = []
    milp_frame = None

    for path in Path(plots_dir).glob('*.png'):
        m = FRAME_NAME_RE.search(path.name)
        if not m:
            continue
        cycle, iteration, tag = int(m.group(1)), int(m.group(2)), m.group(3)
        if tag == 'milp_final':
            milp_frame = path
        else:
            # 'final' (SGD's own closing re-solve) sorts right after the same-numbered
            # regular iteration frame, since it reflects the same iteration count.
            is_final = 1 if tag == 'final' else 0
            iteration_frames.append((cycle, iteration, is_final, path))

    iteration_frames.sort(key=lambda t: t[:3])
    return [t[3] for t in iteration_frames], milp_frame


def _make_title_card(size, text, subtext=None):
    """Dark title-card frame used as a visual break before the MILP closing shot."""
    width, height = size
    img = Image.new('RGB', (width, height), color=(15, 18, 24))
    draw = ImageDraw.Draw(img)

    font = ImageFont.load_default(size=int(height * 0.055))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) / 2, (height - th) / 2 - bbox[1]), text, font=font, fill=(255, 210, 80))

    if subtext:
        sub_font = ImageFont.load_default(size=int(height * 0.028))
        sbbox = draw.textbbox((0, 0), subtext, font=sub_font)
        stw, sth = sbbox[2] - sbbox[0], sbbox[3] - sbbox[1]
        draw.text(((width - stw) / 2, (height - th) / 2 - bbox[1] + th + height * 0.03),
                   subtext, font=sub_font, fill=(200, 200, 200))

    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def build_convergence_video(plots_dir, output_path=None, fps=24, frame_seconds=0.15,
                             title_seconds=2.5, final_seconds=5.0,
                             title_text="WARM START", title_subtext="Initializing MILP cable routing..."):
    """
    Stitches every per-iteration layout plot in plots_dir into an MP4: the SGD
    optimization sequence in chronological order, a title card announcing the MILP
    warm-start refinement, and the MILP final frame held longer as the closing shot.
    """
    plots_dir = Path(plots_dir)
    iteration_frames, milp_frame = _sorted_iteration_frames(plots_dir)

    if not iteration_frames:
        raise FileNotFoundError(f"No iteration plots found in {plots_dir}")

    first = cv2.imread(str(iteration_frames[0]))
    if first is None:
        raise IOError(f"Could not read {iteration_frames[0]}")
    height, width = first.shape[:2]

    if output_path is None:
        output_path = plots_dir.parent / f"{plots_dir.name}_convergence.mp4"
    output_path = Path(output_path)

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"OpenCV could not open a video writer for {output_path}")

    def write_frame(frame, seconds):
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        for _ in range(max(1, round(seconds * fps))):
            writer.write(frame)

    print(f"Writing {len(iteration_frames)} iteration frames...")
    for path in iteration_frames:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[Skipped] Could not read {path}")
            continue
        write_frame(frame, frame_seconds)

    if milp_frame is not None:
        print("Writing title card + MILP warm-start closing frame...")
        write_frame(_make_title_card((width, height), title_text, title_subtext), title_seconds)
        milp_bgr = cv2.imread(str(milp_frame))
        write_frame(milp_bgr, final_seconds)
    else:
        print(f"[Note] No '*_milp_final.png' frame found in {plots_dir} -- "
              f"video ends at the last SGD iteration. Run Warm_Start_Cables.py first "
              f"if you want the MILP closing shot.")

    writer.release()

    n_frames = len(iteration_frames) * max(1, round(frame_seconds * fps))
    if milp_frame is not None:
        n_frames += max(1, round(title_seconds * fps)) + max(1, round(final_seconds * fps))
    print(f"[Exported] Video saved to: {output_path} (~{n_frames / fps:.1f} s, {n_frames} frames @ {fps} fps)")
    return output_path


if __name__ == "__main__":
    RESULTS_DIR = Path("Results_Nash_Run")
    plot_dirs = sorted(p for p in (RESULTS_DIR / "plots").glob("*") if p.is_dir()) \
        if (RESULTS_DIR / "plots").exists() else []

    if not plot_dirs:
        print(f"No plot folders found under {RESULTS_DIR / 'plots'}. "
              f"Run Optimize_farm.py with save_plots=True first.")
    else:
        for pdir in plot_dirs:
            build_convergence_video(pdir)
