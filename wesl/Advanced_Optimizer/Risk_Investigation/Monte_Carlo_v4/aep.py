import time

import numpy as np
import xarray as xr
from py_wake.wind_turbines import WindTurbines
from py_wake.flow_map import XYGrid, Points
from py_wake.site.xrsite import XRSite

import cluster
import turbines
import wake_models

SIM_WD = list(range(0, 360, 30))
SIM_WS = list(range(3, 26, 1))

HOURS_PER_YEAR = 24 * 365
BG_GRID_N = 60      # background-field grid resolution over self's bbox (validated on the real cluster)
BG_MARGIN = 1000.0  # metres of padding around self's bbox for the background grid


def _pywake_turbine(key):
    if key in cluster.REAL_TURBINE_SPEC:
        return turbines.to_pywake_raw(cluster.REAL_TURBINE_SPEC[key])
    return turbines.to_pywake(key)


def _rated_mw(key):
    if key in cluster.REAL_TURBINE_SPEC:
        return cluster.REAL_TURBINE_SPEC[key]["performance"]["rated_power"] / 1e6
    return turbines.rated_mw(key)


def _multi_turbines(farms):
    keys = sorted({f["turbine"] for f in farms})
    wts = [_pywake_turbine(k) for k in keys]
    return WindTurbines.from_WindTurbine_lst(wts), {k: i for i, k in enumerate(keys)}


def evaluate_background_points(site, farms, self_name, model_name=wake_models.ACTIVE_MODEL, n_cpu=1):
    # Alternative to evaluate(): simulates neighbours once, then queries their
    # flow field only at self's turbine positions (via flow_map(Points)) instead
    # of re-running everyone together. Faster per segment but not in a full
    # trial (~1.04x, noise level) and crashes for the turbopark model. Not used
    # in production because of that -- kept for reference. Full writeup and
    # benchmarks: PIPELINE.md and TRIAL_RECORD.md.
    self_farm = next(f for f in farms if f["name"] == self_name)
    neighbours = [f for f in farms if f["name"] != self_name and len(f["x"])]

    self_x, self_y = list(self_farm["x"]), list(self_farm["y"])
    self_wt = _pywake_turbine(self_farm["turbine"])
    self_wts = WindTurbines.from_WindTurbine_lst([self_wt])
    self_H = self_wt.hub_height()

    t0 = time.time()
    wfm_self_alone = wake_models.MODELS[model_name](site, self_wts)
    t_build = time.time() - t0

    t0 = time.time()
    sim_isolated = wfm_self_alone(self_x, self_y, type=0, wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)
    isolated_aep = float(sim_isolated.aep().sum().values)
    t_isolated = time.time() - t0

    if not neighbours:
        return dict(model=model_name, isolated_aep_gwh=isolated_aep, active_aep_gwh=isolated_aep,
                    loss_pct=0.0, n_active_turbines=len(self_x),
                    t_build_s=t_build, t_isolated_s=t_isolated, t_active_s=0.0,
                    t_neighbours_s=0.0, t_points_s=0.0, t_self_s=0.0)

    t0 = time.time()
    nb_keys = sorted({f["turbine"] for f in neighbours})
    nb_type_by_key = {k: i for i, k in enumerate(nb_keys)}
    nb_wts = WindTurbines.from_WindTurbine_lst([_pywake_turbine(k) for k in nb_keys])
    nb_x = [xi for f in neighbours for xi in f["x"]]
    nb_y = [yi for f in neighbours for yi in f["y"]]
    nb_t = [nb_type_by_key[f["turbine"]] for f in neighbours for _ in f["x"]]
    wfm_nb = wake_models.MODELS[model_name](site, nb_wts)
    sim_nb = wfm_nb(nb_x, nb_y, type=nb_t, wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)
    t_neighbours = time.time() - t0

    t0 = time.time()
    pts = Points(np.array(self_x), np.array(self_y), np.full(len(self_x), self_H))
    flow_pts = sim_nb.flow_map(pts, wd=SIM_WD, ws=SIM_WS)
    t_points = time.time() - t0

    t0 = time.time()
    freq_vars = {k: v for k, v in site.ds.data_vars.items()
                 if k in ("Weibull_A", "Weibull_k", "Sector_frequency", "P")}
    ws_eff = flow_pts.WS_eff.transpose("i", "wd", "ws").values
    # A background field can show a local speedup ahead of a big neighbour farm
    # (blockage effect) -- clip it at free-stream so it never reads as faster-than-wind
    # inflow. Doesn't fix turbopark's crash mentioned above, just the ws sanity.
    ws_eff = np.minimum(ws_eff, np.asarray(SIM_WS, dtype=float)[None, None, :])
    bg_ds = xr.Dataset(
        data_vars=dict(WS=(("i", "wd", "ws"), ws_eff),
                       TI=(("i", "wd", "ws"), flow_pts.TI_eff.transpose("i", "wd", "ws").values),
                       **freq_vars),
        coords=dict(i=np.arange(len(self_x)), wd=SIM_WD, ws=SIM_WS))
    bg_site = XRSite(bg_ds)
    wfm_self_bg = wake_models.MODELS[model_name](bg_site, self_wts)
    sim_self_bg = wfm_self_bg(self_x, self_y, type=0, wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)
    active_aep = float(sim_self_bg.aep().sum().values)
    t_self = time.time() - t0

    n_active_turbines = len(self_x) + len(nb_x)
    loss_pct = (isolated_aep - active_aep) / isolated_aep * 100 if isolated_aep else 0.0
    return dict(model=model_name, isolated_aep_gwh=isolated_aep, active_aep_gwh=active_aep,
                loss_pct=loss_pct, n_active_turbines=n_active_turbines,
                t_build_s=t_build, t_isolated_s=t_isolated, t_active_s=t_neighbours + t_points + t_self,
                t_neighbours_s=t_neighbours, t_points_s=t_points, t_self_s=t_self)


def evaluate_background_field(site, farms, self_name, model_name=wake_models.ACTIVE_MODEL, n_cpu=1,
                               bg_grid_n=BG_GRID_N, bg_margin=BG_MARGIN):
    # Older version of evaluate_background_points() above: builds the neighbours'
    # background field on a grid_n x grid_n mesh instead of querying it directly
    # at self's turbines. Grid cost dominates and it ends up slower than the
    # simple merged evaluate() below, so it's not used -- kept for history.
    # See PIPELINE.md.
    self_farm = next(f for f in farms if f["name"] == self_name)
    neighbours = [f for f in farms if f["name"] != self_name and len(f["x"])]

    self_x, self_y = list(self_farm["x"]), list(self_farm["y"])
    self_wt = _pywake_turbine(self_farm["turbine"])
    self_wts = WindTurbines.from_WindTurbine_lst([self_wt])

    t0 = time.time()
    wfm_self_alone = wake_models.MODELS[model_name](site, self_wts)
    t_build = time.time() - t0

    t0 = time.time()
    sim_isolated = wfm_self_alone(self_x, self_y, type=0, wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)
    isolated_aep = float(sim_isolated.aep().sum().values)
    t_isolated = time.time() - t0

    if not neighbours:
        return dict(model=model_name, isolated_aep_gwh=isolated_aep, active_aep_gwh=isolated_aep,
                    loss_pct=0.0, n_active_turbines=len(self_x),
                    t_build_s=t_build, t_isolated_s=t_isolated, t_active_s=0.0,
                    t_background_s=0.0, t_self_s=0.0)

    t0 = time.time()
    nb_keys = sorted({f["turbine"] for f in neighbours})
    nb_type_by_key = {k: i for i, k in enumerate(nb_keys)}
    nb_wts = WindTurbines.from_WindTurbine_lst([_pywake_turbine(k) for k in nb_keys])
    nb_x = [xi for f in neighbours for xi in f["x"]]
    nb_y = [yi for f in neighbours for yi in f["y"]]
    nb_t = [nb_type_by_key[f["turbine"]] for f in neighbours for _ in f["x"]]

    wfm_nb = wake_models.MODELS[model_name](site, nb_wts)
    sim_nb = wfm_nb(nb_x, nb_y, type=nb_t, wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)

    self_H = self_wt.hub_height()
    xmin, xmax = min(self_x) - bg_margin, max(self_x) + bg_margin
    ymin, ymax = min(self_y) - bg_margin, max(self_y) + bg_margin
    grid = XYGrid(x=np.linspace(xmin, xmax, bg_grid_n), y=np.linspace(ymin, ymax, bg_grid_n), h=self_H)
    flowbox = sim_nb.flow_map(grid, wd=SIM_WD, ws=SIM_WS)
    combined_site = XRSite.from_flow_box(xr.Dataset(flowbox).copy(deep=True))
    t_background = time.time() - t0

    t0 = time.time()
    wfm_self_combined = wake_models.MODELS[model_name](combined_site, self_wts)
    sim_self = wfm_self_combined(self_x, self_y, type=0, wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)
    power_self = np.asarray(self_wt.power(sim_self.WS_eff.values))
    # Use sim_isolated's P (real site frequency at self's positions), not the
    # background field's own P, which assumes a different setup and would be wrong here.
    active_aep = float((power_self * sim_isolated.P.values * HOURS_PER_YEAR * 1e-9).sum())
    t_self = time.time() - t0

    n_active_turbines = len(self_x) + len(nb_x)
    loss_pct = (isolated_aep - active_aep) / isolated_aep * 100 if isolated_aep else 0.0
    return dict(model=model_name, isolated_aep_gwh=isolated_aep, active_aep_gwh=active_aep,
                loss_pct=loss_pct, n_active_turbines=n_active_turbines,
                t_build_s=t_build, t_isolated_s=t_isolated, t_active_s=t_background + t_self,
                t_background_s=t_background, t_self_s=t_self)


def evaluate(site, farms, self_name, model_name=wake_models.ACTIVE_MODEL, n_cpu=1):
    # Production default: puts every active turbine (self + neighbours) into
    # one PyWake call. Simple, works with any wake model. The alternatives
    # above were tried to speed this up but weren't worth it -- see PIPELINE.md.
    t0 = time.time()
    multi_turbines, type_by_key = _multi_turbines(farms)
    wfm = wake_models.MODELS[model_name](site, multi_turbines)
    t_build = time.time() - t0

    self_farm = next(f for f in farms if f["name"] == self_name)
    self_type = type_by_key[self_farm["turbine"]]
    t0 = time.time()
    sim_isolated = wfm(list(self_farm["x"]), list(self_farm["y"]),
                        type=[self_type] * len(self_farm["x"]), wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)
    isolated_aep = float(sim_isolated.aep().sum().values)
    t_isolated = time.time() - t0

    others = [f for f in farms if f["name"] != self_name and len(f["x"])]
    x = list(self_farm["x"]) + [xi for f in others for xi in f["x"]]
    y = list(self_farm["y"]) + [yi for f in others for yi in f["y"]]
    t = [self_type] * len(self_farm["x"]) + [type_by_key[f["turbine"]] for f in others for _ in f["x"]]
    n_self = len(self_farm["x"])

    t0 = time.time()
    sim_active = wfm(x, y, type=t, wd=SIM_WD, ws=SIM_WS, n_cpu=n_cpu)
    # aep() covers every turbine in this call, so world_aep is free -- no extra simulation
    active_aep_per_turbine = sim_active.aep()
    active_aep = float(active_aep_per_turbine.isel(wt=slice(0, n_self)).sum().values)
    world_aep = float(active_aep_per_turbine.sum().values)
    t_active = time.time() - t0

    loss_pct = (isolated_aep - active_aep) / isolated_aep * 100 if isolated_aep else 0.0
    active_mw = n_self * _rated_mw(self_farm["turbine"]) + sum(len(f["x"]) * _rated_mw(f["turbine"]) for f in others)
    return dict(model=model_name, isolated_aep_gwh=isolated_aep, active_aep_gwh=active_aep,
                world_aep_gwh=world_aep, loss_pct=loss_pct, n_active_turbines=len(x), active_mw=active_mw,
                t_build_s=t_build, t_isolated_s=t_isolated, t_active_s=t_active)
