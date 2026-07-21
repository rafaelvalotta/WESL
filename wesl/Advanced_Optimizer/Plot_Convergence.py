# Developed by Bruno de Miranda Boer
# University of Massachusetts Boston, WESLab
# Supervised by Professor Rafael Vallota Rodrigues

import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Consistent with Components/plotting.py's palette
LCOE_COLOR = '#1f4e8c'
AEP_COLOR = '#d97b29'
MILP_COLOR = 'crimson'


def _load_convergence(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    history = data['history']
    return {
        'farm_name': data['farm_name'],
        'cycle': data['cycle'],
        'iteration': [h['iteration'] for h in history],
        'elapsed_s': [h['elapsed_s'] for h in history],
        'lcoe_eur_mwh': [h['lcoe_eur_mwh'] for h in history],
        'aep_mwh': [h['aep_mwh'] for h in history],
    }


def _find_milp_result(convergence_path):
    """Looks for a matching checkpoint_<farm>_cycle_<cycle>_milp.pkl next to the
    convergence pickle, so its final LCOE can be overlaid as a reference line."""
    convergence_path = Path(convergence_path)
    m = convergence_path.stem.replace('convergence_', '', 1)
    milp_path = convergence_path.with_name(f"checkpoint_{m}_milp.pkl")
    if not milp_path.exists():
        return None
    with open(milp_path, 'rb') as f:
        milp_data = pickle.load(f)
    return milp_data.get('lcoe_eur_mwh')


def plot_convergence(convergence_paths, output_path=None, show_milp_reference=True):
    """
    Reads one or more convergence-history pickles (written by Optimize_farm.py, one per
    farm/cycle) and renders a two-panel figure: deterministic LCOE and AEP vs iteration
    (left) and vs wall-clock time (right). Both curves come from the driver's noise-free
    full-wind-rose re-evaluation, not the SGD driver's own stochastic mini-batch estimate.

    If a matching '<checkpoint>_milp.pkl' warm-start result is found next to a
    convergence pickle, its final LCOE is drawn as a reference line for comparison.
    """
    if isinstance(convergence_paths, (str, Path)):
        convergence_paths = [convergence_paths]
    runs = [_load_convergence(p) for p in convergence_paths]

    fig, (ax_it, ax_t) = plt.subplots(1, 2, figsize=(15, 6))
    ax_it_aep = ax_it.twinx()
    ax_t_aep = ax_t.twinx()

    use_minutes = max(r['elapsed_s'][-1] for r in runs) > 150
    time_scale = 1 / 60.0 if use_minutes else 1.0
    time_unit = "min" if use_minutes else "s"

    n_runs = len(runs)
    for i, (run, path) in enumerate(zip(runs, convergence_paths)):
        label = f"{run['farm_name']} (cycle {run['cycle']})" if n_runs > 1 else None
        lcoe_style = dict(color=LCOE_COLOR, linewidth=2.0, alpha=1.0 - 0.5 * i / max(n_runs - 1, 1))
        aep_style = dict(color=AEP_COLOR, linewidth=1.4, linestyle='--', alpha=0.8 - 0.4 * i / max(n_runs - 1, 1))

        ax_it.plot(run['iteration'], run['lcoe_eur_mwh'], label=label or 'LCOE (deterministic)', **lcoe_style)
        ax_it_aep.plot(run['iteration'], run['aep_mwh'], **aep_style)

        t = [s * time_scale for s in run['elapsed_s']]
        ax_t.plot(t, run['lcoe_eur_mwh'], **lcoe_style)
        ax_t_aep.plot(t, run['aep_mwh'], **aep_style)

        if show_milp_reference:
            milp_lcoe = _find_milp_result(path)
            if milp_lcoe is not None:
                for ax in (ax_it, ax_t):
                    ax.axhline(milp_lcoe, color=MILP_COLOR, linestyle=':', linewidth=1.6, zorder=1)
                ax_t.annotate(f"MILP warm-start: {milp_lcoe:,.2f} EUR/MWh",
                               xy=(1, milp_lcoe), xycoords=('axes fraction', 'data'),
                               xytext=(-6, 6), textcoords='offset points',
                               ha='right', fontsize=9, color=MILP_COLOR, fontweight='bold')

    for ax, ax_aep, xlabel in [(ax_it, ax_it_aep, 'Iteration'), (ax_t, ax_t_aep, f'Elapsed Time ({time_unit})')]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel('LCOE (EUR/MWh)', color=LCOE_COLOR)
        ax.tick_params(axis='y', labelcolor=LCOE_COLOR)
        ax_aep.set_ylabel('AEP (MWh)', color=AEP_COLOR)
        ax_aep.tick_params(axis='y', labelcolor=AEP_COLOR)
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.spines['top'].set_visible(False)
        ax_aep.spines['top'].set_visible(False)

    ax_it.set_title('Convergence vs. Iteration', fontsize=12, fontweight='bold')
    ax_t.set_title('Convergence vs. Wall-Clock Time', fontsize=12, fontweight='bold')

    farm_titles = ", ".join(sorted({r['farm_name'].upper() for r in runs}))
    fig.suptitle(f"Deterministic LCOE / AEP Convergence — {farm_titles}", fontsize=14, fontweight='bold')

    legend_handles = [
        plt.Line2D([0], [0], color=LCOE_COLOR, linewidth=2.0, label='LCOE (deterministic)'),
        plt.Line2D([0], [0], color=AEP_COLOR, linewidth=1.4, linestyle='--', label='AEP (deterministic)'),
    ]
    if show_milp_reference and any(_find_milp_result(p) is not None for p in convergence_paths):
        legend_handles.append(plt.Line2D([0], [0], color=MILP_COLOR, linestyle=':', linewidth=1.6,
                                          label='MILP warm-start LCOE'))
    if n_runs > 1:
        for run in runs:
            legend_handles.append(plt.Line2D([0], [0], color=LCOE_COLOR,
                                              label=f"{run['farm_name']} (cycle {run['cycle']})"))
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles), bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.04, 1, 1))

    if output_path is None:
        first = Path(convergence_paths[0])
        suffix = "" if n_runs == 1 else "_combined"
        output_path = first.with_name(first.stem + suffix + ".png")
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[Exported] Convergence plot saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    RESULTS_DIR = Path("Results_Nash_Run")
    convergence_files = sorted(RESULTS_DIR.glob("convergence_*_cycle_*.pkl"))

    if not convergence_files:
        print(f"No convergence history found in {RESULTS_DIR}. "
              f"Run Optimize_farm.py with stochastic_mode=True and log_convergence=True first.")
    else:
        for f in convergence_files:
            plot_convergence(f)
        if len(convergence_files) > 1:
            plot_convergence(convergence_files, output_path=RESULTS_DIR / "convergence_all_farms.png")
