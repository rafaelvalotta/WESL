"""Layout QA plot + wake flow-map snapshots for the populated cluster."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from py_wake.flow_map import HorizontalGrid

from config import FIGURES_DIR, FOCUS_FARM, FLOWMAP_WS_SNAPSHOT, FLOWMAP_GRID_RESOLUTION

FLOWMAP_SCENARIOS = [
    ("dominant_240", 240, "Dominant regional wind (240 deg, WSW)"),
    ("north_0", 0, "Wind from due north (0 deg)"),
]


def plot_layout(farms, out_name="cluster_layout_qa"):
    fig, ax = plt.subplots(figsize=(13, 12))
    cmap = colormaps["tab20"].resampled(max(len(farms), 1))

    for i, farm in enumerate(farms):
        if farm["n_turbines"] == 0:
            continue
        kind = "synthetic" if farm["is_synthetic"] else "real"
        marker = "x" if farm["is_synthetic"] else "o"
        ax.scatter(
            np.array(farm["x"]) / 1000, np.array(farm["y"]) / 1000,
            s=16 if farm["is_synthetic"] else 20,
            marker=marker, color=cmap(i),
            edgecolor="none" if farm["is_synthetic"] else "black", linewidth=0.3,
            label=f"{farm['name']} ({kind}, n={farm['n_turbines']})",
        )

    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("US Cluster layout ('o' = real, 'x' = synthetic fill)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    out_path = FIGURES_DIR / f"{out_name}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_flow_maps(aep_result, farms):
    wfm = aep_result["wfm"]
    x_full, y_full, type_full = aep_result["x_full"], aep_result["y_full"], aep_result["type_full"]

    for out_name, wd, title in FLOWMAP_SCENARIOS:
        print(f"[{out_name}] wd={wd}, ws={FLOWMAP_WS_SNAPSHOT}, {len(x_full)} turbines...")
        sim_res = wfm(x_full, y_full, type=type_full, wd=[wd], ws=[FLOWMAP_WS_SNAPSHOT])
        fmap = sim_res.flow_map(grid=HorizontalGrid(resolution=FLOWMAP_GRID_RESOLUTION))

        ws_eff = fmap.WS_eff.squeeze().values
        vmin = max(np.nanpercentile(ws_eff, 1), FLOWMAP_WS_SNAPSHOT * 0.5)
        levels = np.linspace(vmin, FLOWMAP_WS_SNAPSHOT * 1.02, 100)

        fig, ax = plt.subplots(figsize=(14, 13))
        fmap.plot_wake_map(ax=ax, cmap="Blues_r", normalize_with=1000, levels=levels, plot_windturbines=False)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

        for farm in farms:
            if farm["n_turbines"] == 0:
                continue
            if farm["name"] == FOCUS_FARM:
                color = "black"
            elif farm["is_synthetic"]:
                color = "darkorange"
            else:
                color = "teal"
            ax.scatter(np.array(farm["x"]) / 1000, np.array(farm["y"]) / 1000,
                       s=6, color=color, edgecolor="white", linewidth=0.2, zorder=5)

        ax.set_title(title, fontsize=12)
        out_path = FIGURES_DIR / f"wake_flow_map_{out_name}.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")
