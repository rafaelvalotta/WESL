import numpy as np

import windIO
from pathlib import Path
import openmdao.api as om
import matplotlib.pyplot as plt
# from Components.aep import AEPComp


import numpy as np
import openmdao.api as om
import xarray as xr
from py_wake.site import XRSite
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.literature.turbopark import Nygaard_2022  
from py_wake.utils.gradients import autograd  

class AEPComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('n_wt', types=int)
        self.options.declare('system_data', types=dict)
        self.options.declare('site_data', types=dict)
        self.options.declare('stochastic_mode', types=bool, default=False)
        self.options.declare('sample_size', types=int, default=1)
        self.options.declare('x_offset', types=float)
        self.options.declare('y_offset', types=float)
        self.options.declare('neighbor_x', types=np.ndarray)
        self.options.declare('neighbor_y', types=np.ndarray)

    def setup(self):
        n_wt = self.options['n_wt']
        
        self.add_input('x', val=np.zeros(n_wt), units='m')
        self.add_input('y', val=np.zeros(n_wt), units='m')
        self.add_output('aep', val=0.0)

        # Extração dos dados da turbina
        system_data = self.options['system_data']
        turb_info = system_data['wind_farm'][0]['turbines']
        perf_info = turb_info['performance']
        
        p_ws = np.array(perf_info['power_curve']['power_wind_speeds'])
        p = np.array(perf_info['power_curve']['power_values'])
        ct_ws_raw = np.array(perf_info['Ct_curve']['Ct_wind_speeds'])
        ct_val_raw = np.array(perf_info['Ct_curve']['Ct_values'])
        ct = np.interp(p_ws, ct_ws_raw, ct_val_raw)
        
        D = turb_info['rotor_diameter']   
        H = turb_info['hub_height']       

        powerCtFunction = PowerCtTabular(ws=p_ws, power=p, power_unit='W', ct=ct)
        wind_turbine_model = WindTurbine(name="IEA_22MW_Real", diameter=D, hub_height=H, powerCtFunction=powerCtFunction)

        # Montagem do Sítio
        site_dat = self.options['site_data']
        wd_original = site_dat['directions']

        # A TI real varia com a velocidade do vento (TI_org), não com a direção.
        # Guardamos a curva TI(ws) à parte para interpolar sob demanda, em vez de
        # embuti-la como coordenada 'ws' do XRSite -- essa coordenada extra é o que
        # quebra o modo time=True do PyWake (bug de reshape/broadcast confirmado).
        self._ws_TI_curve = np.asarray(site_dat['ws_TI'])
        self._TI_curve = np.asarray(site_dat['TI_org'])
        if self._TI_curve.ndim == 0:
            self._TI_curve = np.full(self._ws_TI_curve.shape, float(self._TI_curve))

        self.pywake_site = XRSite(
            ds=xr.Dataset(
                data_vars={
                    'Sector_frequency': ('wd', site_dat['probabilities']), 
                    'Weibull_A': ('wd', site_dat['weibull_A']), 
                    'Weibull_k': ('wd', site_dat['weibull_k']),
                    'TI': float(np.mean(self._TI_curve)),  # placeholder; sempre sobrescrito via TI= na chamada
                },
                coords={'wd': wd_original}
            )
        )
        self.pywake_site.interp_method = 'linear'
        self.wake_model = Nygaard_2022(self.pywake_site, wind_turbine_model)

        # Grade explícita usada no modo determinístico (rosa cheia), reproduzindo
        # a mesma matriz TI(wd,ws) original, mas passada por chamada via TI= em vez
        # de embutida como coordenada do site.
        self._wd_full = np.asarray(wd_original)
        self._ws_full = self._ws_TI_curve
        self._TI_grid_full = np.tile(self._TI_curve, (len(self._wd_full), 1))
        
        self._current_wd = None
        self._current_ws = None
        self._current_ti = None

        # =============================================================================
        # ANCORAGEM FIXA DO OFFSET GEOGRÁFICO
        # =============================================================================
        # Extrai o ponto mínimo original direto do YAML para ancorar o mapa permanentemente
        # self._x_offset = np.min(system_data['wind_farm'][0]['layouts'][0]['coordinates']['x'])
        # self._y_offset = np.min(system_data['wind_farm'][0]['layouts'][0]['coordinates']['y'])

        self.declare_partials('aep', 'x')
        self.declare_partials('aep', 'y')

    def _sample_metocean(self):
        samps = self.options['sample_size']
        wd_coords = self.pywake_site.ds.wd.values
        freqs = self.pywake_site.ds.Sector_frequency.values
        freqs = freqs / freqs.sum()
        A_vals = self.pywake_site.ds.Weibull_A.values
        k_vals = self.pywake_site.ds.Weibull_k.values

        wd_idx = np.random.choice(np.arange(wd_coords.size), size=samps, p=freqs)
        wd_samp = wd_coords[wd_idx]
        ws_samp = A_vals[wd_idx] * np.random.weibull(k_vals[wd_idx])
        ti_samp = np.interp(ws_samp, self._ws_TI_curve, self._TI_curve)
        return wd_samp, ws_samp, ti_samp

    def compute(self, inputs, outputs):
        # Sem offsets manuais! Passamos direto para o PyWake porque o OpenMDAO 
        # vai entregar o x e o y escalonados de forma invisível.
        wfm = self.wake_model
        # 2. We transform global real inputs to local variables inside the component
        x_local = inputs['x'] - self.options['x_offset']
        y_local = inputs['y'] - self.options['y_offset']
        
        # Neighbors are already processed to local coordinates during system initialization
        xn_local = self.options['neighbor_x'] - self.options['x_offset'] if len(self.options['neighbor_x']) > 0 else np.array([])
        yn_local = self.options['neighbor_y'] - self.options['y_offset'] if len(self.options['neighbor_y']) > 0 else np.array([])
        
        x_full = np.concatenate((x_local, xn_local)) if len(xn_local) > 0 else x_local
        y_full = np.concatenate((y_local, yn_local)) if len(yn_local) > 0 else y_local
            

        if self.options['stochastic_mode']:
            self._current_wd, self._current_ws, self._current_ti = self._sample_metocean()
            sim_res = wfm(x=x_full, y=y_full, wd=self._current_wd, ws=self._current_ws,
                          TI=self._current_ti, time=True, n_cpu=1)
        else:
            self._current_wd, self._current_ws, self._current_ti = None, None, None
            sim_res = wfm(x=x_full, y=y_full, wd=self._wd_full, ws=self._ws_full,
                          TI=self._TI_grid_full, time=False, n_cpu=1)

        if len(xn_local) > 0:
            aep_raw = sim_res.aep().isel(wt=slice(0, len(x_local))).sum().values
        else:
            aep_raw = sim_res.aep().sum().values

        outputs['aep'] = np.abs(float(aep_raw))


    def compute_partials(self, inputs, J):
        wfm = self.wake_model
        n_active = len(inputs['x']) # active turbine count
        
        # 1. Transform global real inputs to local variables for PyWake autograd safety
        x_local = inputs['x'] - self.options['x_offset']
        y_local = inputs['y'] - self.options['y_offset']
        
        # Transform static neighbors to local coordinates as well
        xn_local = self.options['neighbor_x'] - self.options['x_offset'] if len(self.options['neighbor_x']) > 0 else np.array([])
        yn_local = self.options['neighbor_y'] - self.options['y_offset'] if len(self.options['neighbor_y']) > 0 else np.array([])
        
        x_full = np.concatenate((x_local, xn_local)) if len(xn_local) > 0 else x_local
        y_full = np.concatenate((y_local, yn_local)) if len(yn_local) > 0 else y_local

        wd_arg = self._current_wd if self.options['stochastic_mode'] else self._wd_full
        ws_arg = self._current_ws if self.options['stochastic_mode'] else self._ws_full
        ti_arg = self._current_ti if self.options['stochastic_mode'] else self._TI_grid_full
        time_arg = True if self.options['stochastic_mode'] else False

        # IMPORTANTE: wfm.aep()/aep_gradients() sempre somam a AEP de TODAS as
        # turbinas passadas (ativas + vizinhas) -- não existe forma de restringir
        # a soma a um subconjunto. Diferenciar wfm.aep_gradients(...) e depois
        # fatiar [:n_active] contamina o gradiente com d(AEP_vizinha)/dx_ativa,
        # que não é o que compute() calcula (compute() soma só as ativas).
        # Por isso replicamos o corpo interno de wfm.aep() aqui, fatiando as
        # turbinas ativas ANTES da soma, mantendo tudo dentro do grafo do autograd
        # (wfm._run retorna arrays numpy/autograd puros; o objeto SimulationResult
        # de wfm(...) quebra o rastreamento do autograd e não pode ser usado aqui).
        def active_aep_only(x, y):
            res = wfm._run(x, y, h=None, type=0, wd=wd_arg, ws=ws_arg, TI=ti_arg,
                            time=time_arg, n_cpu=1)
            _, _, power_ilk, _, localWind, _ = res
            P_ilk = localWind.P_ilk
            power_active = power_ilk[:n_active]
            P_active = P_ilk[:n_active]
            return (power_active * P_active * 24 * 365 * 1e-9).sum()

        grad_active = autograd(active_aep_only, True, argnum=[0, 1])
        daep_dx, daep_dy = grad_active(x_full, y_full)
        daep_dx = np.array(daep_dx)[:n_active]
        daep_dy = np.array(daep_dy)[:n_active]

        # 2. Return the exact analytical gradients to OpenMDAO's global scale
        J['aep', 'x'] = np.array(daep_dx).reshape(1, -1)
        J['aep', 'y'] = np.array(daep_dy).reshape(1, -1)

# SCRIPT_DIR = Path(__file__).parent.parent
# SYSTEM_YAML_PATH = SCRIPT_DIR / "Data" / "vineyard_revolution_system.yaml"

# print("=== 1. LOADING REAL GEOGRAPHIC DATA VIA WINDIO ===")
# system_dat = windIO.load_yaml(SYSTEM_YAML_PATH)

# wind_resource = system_dat['site']['energy_resource']['wind_resource']
# site_resource_dict = {
#     'directions': np.array(wind_resource['wind_direction']),
#     'probabilities': np.array(wind_resource['sector_probability']['data']),
#     'weibull_A': np.array(wind_resource['weibull_a']['data']),
#     'weibull_k': np.array(wind_resource['weibull_k']['data']),
#     'ws_TI': np.array(wind_resource['wind_speed']),
#     'TI_org': wind_resource['turbulence_intensity']['data']
# }
# site_resource_dict['probabilities'] /= site_resource_dict['probabilities'].sum()

# # Extract full real UTM scale coordinates (millions of meters)
# x_real_active_full = np.array(system_dat['wind_farm'][0]['layouts'][0]['coordinates']['x'])
# y_real_active_full = np.array(system_dat['wind_farm'][0]['layouts'][0]['coordinates']['y'])

# neighbor_x_full = np.array(system_dat['wind_farm'][1]['layouts'][0]['coordinates']['x'])
# neighbor_y_full = np.array(system_dat['wind_farm'][1]['layouts'][0]['coordinates']['y'])

# x_real_active = x_real_active_full#[:10] # 10 active turbines (real scale)
# y_real_active = y_real_active_full#[:10]

# neighbor_x = neighbor_x_full#[:10] # 10 neighbor turbines (real scale)
# neighbor_y = neighbor_y_full#[:10]

# N_WT = len(x_real_active)

# # Compute the static anchor offsets
# x_offset = x_real_active.min()
# y_offset = y_real_active.min()

# print(f"[Success] Light sub-grid generated for test.")
# print(f"Active Farm: {N_WT} turbines. Neighbor Farm: {len(neighbor_x)} turbines.")

# # Slice wind rose
# wd_sliced = site_resource_dict['directions']#[::45]
# prob_sliced = site_resource_dict['probabilities']#[::45]
# prob_sliced /= prob_sliced.sum()

# ws_step = 4
# ws_sliced = site_resource_dict['ws_TI'][::ws_step]
# ti_sliced = site_resource_dict['TI_org'][::ws_step] if np.ndim(site_resource_dict['TI_org']) > 0 else site_resource_dict['TI_org']

# site_resource_VALIDATION = {
#     'directions': wd_sliced,
#     'probabilities': prob_sliced,
#     'weibull_A': site_resource_dict['weibull_A'][::45],
#     'weibull_k': site_resource_dict['weibull_k'][::45],
#     'ws_TI': ws_sliced,
#     'TI_org': ti_sliced
# }

# print("\n=== 2. INSTANTIATING AND INTEGRATING AEP COMPONENT ===")
# prob = om.Problem()

# # Pass real scale neighbors and static offsets as options
# comp = AEPComp(
#     n_wt=N_WT,
#     system_data=system_dat,
#     site_data=site_resource_dict,
#     stochastic_mode=False,
#     sample_size=100,             
#     x_offset=x_offset,
#     y_offset=y_offset,
#     neighbor_x=neighbor_x, # real coordinates
#     neighbor_y=neighbor_y  # real coordinates
# )

# prob.model.add_subsystem('aep_comp', comp, promotes=['*'])
# prob.setup()

# # Feed OpenMDAO with full real UTM scale coordinates
# prob.set_val('x', x_real_active)
# prob.set_val('y', y_real_active)

# print("Calculating deterministic continuous wake flow...")
# prob.run_model()

# aep_final = prob.get_val('aep')[0]
# print(f"--> [Success] Sampled Deterministic AEP: {aep_final:.4f} MWh")

# print("\n=== 3. VALIDATING COMPONENT PARTIALS ===")
# prob.check_partials(compact_print=True, method='fd', step=1.0)

# print("\n=== 4. AUTHENTICATING REAL NEIGHBORING IMPACT ===")
# prob_isolated = om.Problem()

# # Isolated component with empty neighbor arrays
# comp_no_neighbor = AEPComp(
#     n_wt=N_WT,
#     system_data=system_dat,
#     site_data=site_resource_dict,  
#     stochastic_mode=True,             
#     sample_size=100,                 
#     x_offset=x_offset,
#     y_offset=y_offset,
#     neighbor_x=np.array([]),
#     neighbor_y=np.array([])
# )

# prob_isolated.model.add_subsystem('aep_no_neighbor', comp_no_neighbor, promotes=['*'])
# prob_isolated.setup()

# # Feed isolated problem with the exact same full real UTM scale coordinates
# prob_isolated.set_val('x', x_real_active)
# prob_isolated.set_val('y', y_real_active)

# prob_isolated.run_model()
# aep_no_neighbor = prob_isolated.get_val('aep')[0]

# loss_mwh = aep_no_neighbor - aep_final
# loss_percentage = (loss_mwh / aep_no_neighbor) * 100

# print(f"AEP with neighboring plant ON:  {aep_final:.4f} MWh")
# print(f"AEP with neighboring plant OFF: {aep_no_neighbor:.4f} MWh")
# print(f"--> [Result] Neighbor plant steals: {loss_mwh:.4f} MWh from active plant.")
# print(f"--> [Real Impact]: {loss_percentage:.2f}% external wake losses confirmed.")

# if loss_mwh > 1e-3:
#     print("[AUTHENTICATED] Success! PyWake correctly couples neighbor flow effects.")
# else:
#     print("[ALERT] Impact is zero. Verify neighbor positions.")


SCRIPT_DIR = Path(__file__).parent.parent
SYSTEM_YAML_PATH = SCRIPT_DIR / "Data" / "vineyard_revolution_system.yaml"

print("=== 1. LOADING REAL GEOGRAPHIC DATA VIA WINDIO ===")
system_dat = windIO.load_yaml(SYSTEM_YAML_PATH)

wind_resource = system_dat['site']['energy_resource']['wind_resource']
site_resource_dict = {
    'directions': np.array(wind_resource['wind_direction']),
    'probabilities': np.array(wind_resource['sector_probability']['data']),
    'weibull_A': np.array(wind_resource['weibull_a']['data']),
    'weibull_k': np.array(wind_resource['weibull_k']['data']),
    'ws_TI': np.array(wind_resource['wind_speed']),
    'TI_org': wind_resource['turbulence_intensity']['data']
}
site_resource_dict['probabilities'] /= site_resource_dict['probabilities'].sum()

# Extração da escala cheia UTM das turbinas
x_real_active = np.array(system_dat['wind_farm'][0]['layouts'][0]['coordinates']['x'])
y_real_active = np.array(system_dat['wind_farm'][0]['layouts'][0]['coordinates']['y'])

neighbor_x = np.array(system_dat['wind_farm'][1]['layouts'][0]['coordinates']['x'])
neighbor_y = np.array(system_dat['wind_farm'][1]['layouts'][0]['coordinates']['y'])

N_WT = len(x_real_active)
x_offset = x_real_active.min()
y_offset = y_real_active.min()

print(f"[Success] Setup carregado para validação estocástica.")
print(f"Active Farm (Vineyard): {N_WT} WT. Neighbor Farm (Revolution): {len(neighbor_x)} WT.")

print("\n=== 2. REFERÊNCIA: DETERMINÍSTICO CHEIO (FULL ROSE) ===")
prob_det = om.Problem()
comp_det = AEPComp(
    n_wt=N_WT,
    system_data=system_dat,
    site_data=site_resource_dict,
    stochastic_mode=False, # Referência determinística real
    sample_size=1,             
    x_offset=x_offset,
    y_offset=y_offset,
    neighbor_x=neighbor_x, 
    neighbor_y=neighbor_y  
)
prob_det.model.add_subsystem('aep_det', comp_det, promotes=['*'])
prob_det.setup()
prob_det.set_val('x', x_real_active)
prob_det.set_val('y', y_real_active)
prob_det.run_model()
aep_det_real = prob_det.get_val('aep')[0]
print(f"--> Deterministic AEP (Rosa Cheia): {aep_det_real:,.2f} MWh")

print("\n=== 3. TESTE DA COMPONENTE EM MODO ESTOCÁSTICO (SAMPLE SIZE = 50) ===")
prob_stoch = om.Problem()
comp_stoch = AEPComp(
    n_wt=N_WT,
    system_data=system_dat,
    site_data=site_resource_dict,
    stochastic_mode=True,  # Ativando a amostragem de Monte Carlo!
    sample_size=150,        # Amostra robusta para mitigar a variância do teste
    x_offset=x_offset,
    y_offset=y_offset,
    neighbor_x=neighbor_x, 
    neighbor_y=neighbor_y  
)
prob_stoch.model.add_subsystem('aep_stoch', comp_stoch, promotes=['*'])
prob_stoch.setup()
prob_stoch.set_val('x', x_real_active)
prob_stoch.set_val('y', y_real_active)

# Vamos rodar 3 vezes seguidas para ver a convergência estatística da escala corrigida!
print("Avaliando convergência da escala estocástica por amostragem...")
for i in range(1, 5):
    prob_stoch.run_model()
    print(f"   Amostra SGD Iter {i} - AEP Escalada Estimada: {prob_stoch.get_val('aep')[0]:,.2f} MWh")


import types
import numpy as np

print("\n=== VALIDANDO GRADIENTES NO MODO ESTOCÁSTICO (time=True) ===")

# 1. Congela a amostra atmosférica uma única vez
wd_fixa, ws_fixa, ti_fixa = comp_stoch._sample_metocean()

def _fake_sample(self):
    return wd_fixa, ws_fixa, ti_fixa

comp_stoch._sample_metocean = types.MethodType(_fake_sample, comp_stoch)

prob_stoch.run_model()
aep_a = prob_stoch.get_val('aep')[0]
prob_stoch.run_model()
aep_b = prob_stoch.get_val('aep')[0]
print(f"AEP call 1: {aep_a:.6f}  |  AEP call 2: {aep_b:.6f}  (devem ser idênticos)")
assert aep_a == aep_b, "Congelamento da amostra falhou"

# 2. Varredura de step, olhando o VETOR INTEIRO de gradientes (todas as
#    turbinas), não só a linha-resumo do compact_print -- que pode trocar
#    de turbina "pior" a cada rodada e enganar a leitura.
print("\n=== VARREDURA DE STEP (vetor completo, mesma amostra em todas as rodadas) ===")
for step in [10.0, 1.0, 0.1, 0.01]:
    data = prob_stoch.check_partials(method='fd', step=step, out_stream=None)
    for comp_name, ofwrt in data.items():
        for (of, wrt), vals in ofwrt.items():
            J_fwd = np.asarray(vals['J_fwd']).flatten()
            J_fd = np.asarray(vals['J_fd']).flatten()
            abs_err = np.abs(J_fwd - J_fd)
            worst_idx = int(np.argmax(abs_err))
            print(f"step={step:6.2f}  {of} wrt {wrt}: max_abs_err={abs_err.max():.3e}"
                  f"  (pior turbina: idx={worst_idx}, calc={J_fwd[worst_idx]:.5f}, fd={J_fd[worst_idx]:.5f})")

# # print("\n=== 4. AGORA SIM: VALIDANDO OS GRADIENTES DO ESTOCÁSTICO ===")
# # # Para checar os parciais analíticos do Autograd contra o FD sem o ruído do vento mudar, 
# # # nós desligamos temporariamente o modo estocástico apenas durante o check_partials, 
# # # garantindo que o grafo de derivadas do PyWake esteja correto.
# # prob_stoch.model.aep_stoch.options['stochastic_mode'] = False
# # prob_stoch.check_partials(compact_print=True, method='fd', step=1e-3)

# SCRIPT_DIR = Path(__file__).parent.parent
# SYSTEM_YAML_PATH = SCRIPT_DIR / "Data" / "vineyard_revolution_system.yaml"

# print("=== 1. LOADING REAL GEOGRAPHIC DATA VIA WINDIO ===")
# system_dat = windIO.load_yaml(SYSTEM_YAML_PATH)

# wind_resource = system_dat['site']['energy_resource']['wind_resource']
# site_resource_dict = {
#     'directions': np.array(wind_resource['wind_direction']),
#     'probabilities': np.array(wind_resource['sector_probability']['data']),
#     'weibull_A': np.array(wind_resource['weibull_a']['data']),
#     'weibull_k': np.array(wind_resource['weibull_k']['data']),
#     'ws_TI': np.array(wind_resource['wind_speed']),
#     'TI_org': wind_resource['turbulence_intensity']['data']
# }
# site_resource_dict['probabilities'] /= site_resource_dict['probabilities'].sum()

# x_real_active = np.array(system_dat['wind_farm'][0]['layouts'][0]['coordinates']['x'])  # Vineyard
# y_real_active = np.array(system_dat['wind_farm'][0]['layouts'][0]['coordinates']['y'])
# neighbor_x = np.array(system_dat['wind_farm'][1]['layouts'][0]['coordinates']['x'])       # Revolution
# neighbor_y = np.array(system_dat['wind_farm'][1]['layouts'][0]['coordinates']['y'])

# N_WT = len(x_real_active)
# x_offset = x_real_active.min()
# y_offset = y_real_active.min()

# print(f"Active Farm (Vineyard): {N_WT} turbines. Neighbor Farm (Revolution): {len(neighbor_x)} turbines.")

# # =====================================================================
# # PARTE A: DEBUG DE GEOMETRIA E CONVENÇÃO DE DIREÇÃO
# # =====================================================================
# print("\n=== 2. DEBUG: GEOMETRIA E ROSA DOS VENTOS ===")

# active_centroid = np.array([x_real_active.mean(), y_real_active.mean()])
# neigh_centroid = np.array([neighbor_x.mean(), neighbor_y.mean()])

# dx = neigh_centroid[0] - active_centroid[0]   # +x assumido Leste (UTM)
# dy = neigh_centroid[1] - active_centroid[1]   # +y assumido Norte (UTM)
# dist_km = np.hypot(dx, dy) / 1000.0

# # Bearing meteorológico (convenção "de onde vem o vento") que conecta
# # Revolution -> Vineyard: é o bearing, visto de Vineyard, apontando para Revolution.
# bearing_neighbor_impact = (np.degrees(np.arctan2(dx, dy))) % 360

# print(f"Centroide Vineyard (ativa):   x={active_centroid[0]:,.1f}  y={active_centroid[1]:,.1f}")
# print(f"Centroide Revolution (vizinha): x={neigh_centroid[0]:,.1f}  y={neigh_centroid[1]:,.1f}")
# print(f"Distancia entre centroides: {dist_km:.2f} km")
# print(f"Bearing meteorologico (direcao de vento que leva a esteira de Revolution ate Vineyard): {bearing_neighbor_impact:.1f} graus")
# print("  (convencao: 0=Norte, 90=Leste, 180=Sul, 270=Oeste, sentido horario)")

# print("\n--- Direções cadastradas no YAML, ordenadas por probabilidade (top 10) ---")
# dirs = site_resource_dict['directions']
# probs = site_resource_dict['probabilities']
# order = np.argsort(-probs)
# for i in order[:10]:
#     marker = ""
#     ang_diff = min(abs(dirs[i] - bearing_neighbor_impact), 360 - abs(dirs[i] - bearing_neighbor_impact))
#     if ang_diff <= 30:
#         marker = "   <-- dentro do setor de impacto (±30°)"
#     print(f"  wd={dirs[i]:6.1f}°   prob={probs[i]*100:5.2f}%{marker}")

# prob_in_sector = probs[np.minimum(np.abs(dirs - bearing_neighbor_impact),
#                                    360 - np.abs(dirs - bearing_neighbor_impact)) <= 30].sum()
# print(f"\nProbabilidade total de vento vindo do setor de impacto (±30° em torno de {bearing_neighbor_impact:.0f}°): {prob_in_sector*100:.2f}%")
# print("Se esse numero for pequeno, o vizinho so raramente fica a montante -> perda anual pequena faz sentido.")
# print("Se for grande (>15-20%) e mesmo assim a perda anual sair baixa, verifique a convenção de direção do YAML.")

# # =====================================================================
# # PARTE B: TESTE DE SETOR ISOLADO (só direções de impacto)
# # =====================================================================
# print("\n=== 3. TESTE DE SETOR ISOLADO (SÓ VENTOS DO SETOR DE IMPACTO) ===")

# sector_mask = np.minimum(np.abs(dirs - bearing_neighbor_impact),
#                           360 - np.abs(dirs - bearing_neighbor_impact)) <= 30

# if sector_mask.sum() == 0:
#     print("[ALERTA] Nenhum bin de direção caiu dentro do setor -- afrouxe a tolerância (ex: 45°) e rode de novo.")
# else:
#     site_sector = {
#         'directions': dirs[sector_mask],
#         'probabilities': probs[sector_mask] / probs[sector_mask].sum(),  # renormalizado dentro do setor
#         'weibull_A': site_resource_dict['weibull_A'][sector_mask],
#         'weibull_k': site_resource_dict['weibull_k'][sector_mask],
#         'ws_TI': site_resource_dict['ws_TI'],      # curva de TI(ws) não depende de direção
#         'TI_org': site_resource_dict['TI_org']
#     }
#     print(f"Bins de direção usados no setor: {sector_mask.sum()} (de {len(dirs)} totais)")

#     # --- com vizinho, só nesse setor ---
#     prob_with = om.Problem()
#     comp_with = AEPComp(
#         n_wt=N_WT, system_data=system_dat, site_data=site_sector,
#         stochastic_mode=False, sample_size=1,
#         x_offset=x_offset, y_offset=y_offset,
#         neighbor_x=neighbor_x, neighbor_y=neighbor_y
#     )
#     prob_with.model.add_subsystem('aep_with', comp_with, promotes=['*'])
#     prob_with.setup()
#     prob_with.set_val('x', x_real_active)
#     prob_with.set_val('y', y_real_active)
#     prob_with.run_model()
#     aep_with_sector = prob_with.get_val('aep')[0]

#     # --- sem vizinho, mesmo setor ---
#     prob_without = om.Problem()
#     comp_without = AEPComp(
#         n_wt=N_WT, system_data=system_dat, site_data=site_sector,
#         stochastic_mode=False, sample_size=1,
#         x_offset=x_offset, y_offset=y_offset,
#         neighbor_x=np.array([]), neighbor_y=np.array([])
#     )
#     prob_without.model.add_subsystem('aep_without', comp_without, promotes=['*'])
#     prob_without.setup()
#     prob_without.set_val('x', x_real_active)
#     prob_without.set_val('y', y_real_active)
#     prob_without.run_model()
#     aep_without_sector = prob_without.get_val('aep')[0]

#     loss_sector_mwh = aep_without_sector - aep_with_sector
#     loss_sector_pct = (loss_sector_mwh / aep_without_sector) * 100

#     print(f"\nAEP (só setor de impacto) COM Revolution:  {aep_with_sector:,.4f} MWh-eq")
#     print(f"AEP (só setor de impacto) SEM Revolution:  {aep_without_sector:,.4f} MWh-eq")
#     print(f"--> Perda por esteira, restrita ao setor de impacto: {loss_sector_mwh:,.4f} MWh-eq ({loss_sector_pct:.2f}%)")
#     print()
#     print("Comparação esperada: esse percentual deve ser BEM maior que o 0.60% da AEP anual completa,")
#     print("já que aqui isolamos só as direções onde Revolution realmente fica a montante de Vineyard.")


# =====================================================================
# BLOCO COMPLEMENTAR: GERAÇÃO DOS RESULTADOS REAIS E GRÁFICO EXECUTIVO
# =====================================================================
print("\n=== 5. GENERATING PRODUCTION DATA FOR REUNION PLOTS ===")

# --- 5.1. CÁLCULO DO IMPACTO ANUAL (ROSA COMPLETA DE VENTO) ---
# AEP com vizinho já calculado no seu script como 'aep_det_real'
aep_with_neighbor_annual = float(aep_det_real)

# Para pegar o caso isolado anual (sem vizinho), instanciamos rapidamente um problema limpo
prob_det_no_neighbor = om.Problem()
comp_det_no_neighbor = AEPComp(
    n_wt=N_WT, system_data=system_dat, site_data=site_resource_dict,
    stochastic_mode=False, sample_size=1,
    x_offset=x_offset, y_offset=y_offset,
    neighbor_x=np.array([]), neighbor_y=np.array([])
)
prob_det_no_neighbor.model.add_subsystem('aep_det_isolated', comp_det_no_neighbor, promotes=['*'])
prob_det_no_neighbor.setup()
prob_det_no_neighbor.set_val('x', x_real_active)
prob_det_no_neighbor.set_val('y', y_real_active)
prob_det_no_neighbor.run_model()
aep_no_neighbor_annual = float(prob_det_no_neighbor.get_val('aep')[0])

annual_loss_pct = ((aep_no_neighbor_annual - aep_with_neighbor_annual) / aep_no_neighbor_annual) * 100


# --- 5.2. CÁLCULO DO IMPACTO GEOMÉTRICO (SETOR DE IMPACTO SELECIONADO) ---
# Calculando dinamicamente o bearing em cima do seu bloco de geometria
active_centroid = np.array([x_real_active.mean(), y_real_active.mean()])
neigh_centroid = np.array([neighbor_x.mean(), neighbor_y.mean()])
dx = neigh_centroid[0] - active_centroid[0]
dy = neigh_centroid[1] - active_centroid[1]
bearing_neighbor_impact = (np.degrees(np.arctan2(dx, dy))) % 360

dirs = site_resource_dict['directions']
probs = site_resource_dict['probabilities']

# Máscara estrita de alinhamento físico (+/- 30 graus da direção crítica)
sector_mask = np.minimum(np.abs(dirs - bearing_neighbor_impact), 
                         360 - np.abs(dirs - bearing_neighbor_impact)) <= 30.0

site_sector = {
    'directions': dirs[sector_mask],
    'probabilities': probs[sector_mask] / probs[sector_mask].sum(), # Renormalizado
    'weibull_A': site_resource_dict['weibull_A'][sector_mask],
    'weibull_k': site_resource_dict['weibull_k'][sector_mask],
    'ws_TI': site_resource_dict['ws_TI'],
    'TI_org': site_resource_dict['TI_org']
}

# Simulação Setorizada COM vizinho
prob_with_sect = om.Problem()
comp_with_sect = AEPComp(
    n_wt=N_WT, system_data=system_dat, site_data=site_sector,
    stochastic_mode=False, sample_size=1,
    x_offset=x_offset, y_offset=y_offset,
    neighbor_x=neighbor_x, neighbor_y=neighbor_y
)
prob_with_sect.model.add_subsystem('aep_with_sect', comp_with_sect, promotes=['*'])
prob_with_sect.setup()
prob_with_sect.set_val('x', x_real_active)
prob_with_sect.set_val('y', y_real_active)
prob_with_sect.run_model()
aep_with_neighbor_sector = float(prob_with_sect.get_val('aep')[0])

# Simulação Setorizada SEM vizinho
prob_without_sect = om.Problem()
comp_without_sect = AEPComp(
    n_wt=N_WT, system_data=system_dat, site_data=site_sector,
    stochastic_mode=False, sample_size=1,
    x_offset=x_offset, y_offset=y_offset,
    neighbor_x=np.array([]), neighbor_y=np.array([])
)
prob_without_sect.model.add_subsystem('aep_without_sect', comp_without_sect, promotes=['*'])
prob_without_sect.setup()
prob_without_sect.set_val('x', x_real_active)
prob_without_sect.set_val('y', y_real_active)
prob_without_sect.run_model()
aep_no_neighbor_sector = float(prob_without_sect.get_val('aep')[0])

sector_loss_pct = ((aep_no_neighbor_sector - aep_with_neighbor_sector) / aep_no_neighbor_sector) * 100


# --- 5.3. PLOT EXECUÇÃO MATPLOTLIB (LAYOUT INGLÊS CORPORATIVO) ---
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

color_baseline = '#1f4e79'  # Azul escuro elegante (Fazenda Isolada)
color_wake = '#d95f02'      # Laranja fosco (Efeito de Esteira Externa)

# Subplot 1: Visão Geral Anualizada
bars1 = ax1.bar(['Isolated\n(Baseline)', 'With Revolution\n(External Wake)'], 
                [aep_no_neighbor_annual, aep_with_neighbor_annual], 
                color=[color_baseline, color_wake], width=0.45, edgecolor='black', alpha=0.9)

ax1.set_title("Annualized Energy Yield (Full Wind Rose)", fontsize=13, fontweight='bold', pad=15)
ax1.set_ylabel("Annual Potential (MWh)", fontsize=11)
ax1.set_ylim(min(aep_no_neighbor_annual, aep_with_neighbor_annual) * 0.85, aep_no_neighbor_annual * 1.1)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + (aep_no_neighbor_annual * 0.02), f"{yval:,.1f}\nMWh", 
             ha='center', va='bottom', color='black', fontweight='bold', fontsize=10)

ax1.annotate(f'-{annual_loss_pct:.2f}% Loss', 
             xy=(0.5, (aep_no_neighbor_annual + aep_with_neighbor_annual)/2), 
             xytext=(0.65, aep_no_neighbor_annual * 1.05),
             arrowprops=dict(facecolor='black', arrowstyle="->", lw=1.2),
             fontsize=11, color='red', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

# Subplot 2: Setor Isolado Crítico de Impacto Direto
bars2 = ax2.bar(['Isolated\n(Baseline)', 'With Revolution\n(External Wake)'], 
                [aep_no_neighbor_sector, aep_with_neighbor_sector], 
                color=[color_baseline, color_wake], width=0.45, edgecolor='black', alpha=0.9)

ax2.set_title(f"Critical Sector Alignment (Wind: {bearing_neighbor_impact:.1f}° $\pm30$°)", fontsize=13, fontweight='bold', pad=15)
ax2.set_ylabel("Sector Energy Contribution (MWh-eq)", fontsize=11)
ax2.set_ylim(min(aep_no_neighbor_sector, aep_with_neighbor_sector) * 0.7, aep_no_neighbor_sector * 1.2)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + (aep_no_neighbor_sector * 0.02), f"{yval:,.1f}\nMWh", 
             ha='center', va='bottom', color='black', fontweight='bold', fontsize=10)

ax2.annotate(f'-{sector_loss_pct:.2f}% Deficit', 
             xy=(0.5, (aep_no_neighbor_sector + aep_with_neighbor_sector)/2), 
             xytext=(0.65, aep_no_neighbor_sector * 1.08),
             arrowprops=dict(facecolor='red', arrowstyle="->", lw=1.5),
             fontsize=12, color='darkred', fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.9))

plt.suptitle("PyWake Coupling Verification: Vineyard Plant vs. Revolution Impact", fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()

# Exporta direto para salvar o PNG pronto para o slide da reunião
output_img = SCRIPT_DIR / "Checks" / "PyWake_Meeting_Impact_Plot.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"--> [Plot Success] Professional figure saved directly at: {output_img}")
plt.show()