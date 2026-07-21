import numpy as np
import windIO
from pathlib import Path
import openmdao.api as om
import xarray as xr
import matplotlib.pyplot as plt

from py_wake.site import XRSite
from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
# --- TROCA DE MODELO: SAINDO DO TURBOPARK PARA O GAUSSIANO ANALÍTICO ---
from py_wake.literature.gaussian_models import Bastankhah_PorteAgel_2014  
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
        ws_original = site_dat['ws_TI']
        ti_matrix = np.zeros((len(wd_original), len(ws_original)))
        for i in range(len(wd_original)):
            ti_matrix[i, :] = site_dat['TI_org']

        self.pywake_site = XRSite(
            ds=xr.Dataset(
                data_vars={
                    'Sector_frequency': ('wd', site_dat['probabilities']), 
                    'Weibull_A': ('wd', site_dat['weibull_A']), 
                    'Weibull_k': ('wd', site_dat['weibull_k']),
                    'TI': (('wd', 'ws'), ti_matrix)
                },
                coords={'wd': wd_original, 'ws': ws_original}
            )
        )
        self.pywake_site.interp_method = 'linear'
        
        # --- ATIVAÇÃO DO MOTOR BASTANKHAH GAUSSIAN ---
        # Configurado com k=0.05 padrão de literatura offshore da DTU
        self.wake_model = Bastankhah_PorteAgel_2014(self.pywake_site, wind_turbine_model, k=0.05)
        
        self._current_wd = None
        self._current_ws = None

        self.declare_partials('aep', 'x')
        self.declare_partials('aep', 'y')

    def _sample_metocean(self):
        samps = self.options['sample_size']
        
        wd_coords = self.pywake_site.ds.wd.values
        ws_coords = self.pywake_site.ds.ws.values
        
        # Limites definidos pelo seu XRSite
        ws_min, ws_max = ws_coords.min(), ws_coords.max()
        
        freqs = self.pywake_site.ds.Sector_frequency.values
        freqs /= freqs.sum() 
        
        # 1. Sorteia as direções
        wd_idx = np.random.choice(np.arange(wd_coords.size), size=samps, p=freqs)
        wd_sampled = wd_coords[wd_idx]
        
        # 2. Extrai Weibull
        As = self.pywake_site.ds.Weibull_A.values[wd_idx]
        ks = self.pywake_site.ds.Weibull_k.values[wd_idx]
        
        # 3. Sorteia e BLINDA a velocidade para não sair do intervalo do sítio
        ws_raw = As * np.random.weibull(ks)
        ws_sampled = np.clip(ws_raw, ws_min, ws_max) # <--- AQUI A BLINDAGEM!
        
        return wd_sampled, ws_sampled

    def compute(self, inputs, outputs):
        wfm = self.wake_model
        x_local = inputs['x'] - self.options['x_offset']
        y_local = inputs['y'] - self.options['y_offset']
        
        xn_local = self.options['neighbor_x'] - self.options['x_offset'] if len(self.options['neighbor_x']) > 0 else np.array([])
        yn_local = self.options['neighbor_y'] - self.options['y_offset'] if len(self.options['neighbor_y']) > 0 else np.array([])
        
        x_full = np.concatenate((x_local, xn_local)) if len(xn_local) > 0 else x_local
        y_full = np.concatenate((y_local, yn_local)) if len(yn_local) > 0 else y_local

        if self.options['stochastic_mode']:
            self._current_wd, self._current_ws = self._sample_metocean()
            sim_res = wfm(x=x_full, y=y_full, wd=self._current_wd, ws=self._current_ws, time=False, n_cpu=1)
            
            aep_raw = sim_res.aep().sum().values
            
            outputs['aep'] = np.abs(float(aep_raw))
        else:
            self._current_wd, self._current_ws = None, None
            sim_res = wfm(x=x_full, y=y_full, n_cpu=1)
            
            if len(xn_local) > 0:
                aep_raw = sim_res.aep().isel(wt=slice(0, len(x_local))).sum().values
            else:
                aep_raw = sim_res.aep().sum().values

        outputs['aep'] = np.abs(float(aep_raw))

    def compute_partials(self, inputs, J):
        wfm = self.wake_model
        n_active = len(inputs['x']) 
        
        x_local = inputs['x'] - self.options['x_offset']
        y_local = inputs['y'] - self.options['y_offset']
        
        xn_local = self.options['neighbor_x'] - self.options['x_offset'] if len(self.options['neighbor_x']) > 0 else np.array([])
        yn_local = self.options['neighbor_y'] - self.options['y_offset'] if len(self.options['neighbor_y']) > 0 else np.array([])
        
        x_full = np.concatenate((x_local, xn_local)) if len(xn_local) > 0 else x_local
        y_full = np.concatenate((y_local, yn_local)) if len(yn_local) > 0 else y_local

        grad_kwargs = {
            "gradient_method": autograd,
            "wrt_arg": ["x", "y"],
            "x": x_full,
            "y": y_full,
            "n_cpu": 1
        }
        
        if self.options['stochastic_mode']:
            grad_kwargs["wd"] = self._current_wd
            grad_kwargs["ws"] = self._current_ws
            # CORREÇÃO: Passar time=True para casar com as dimensões avaliadas no compute()
            grad_kwargs["time"] = False 
            
            n_total_combinations = len(self.pywake_site.ds.wd.values) * len(self.pywake_site.ds.ws.values)
            fator_escala = n_total_combinations / self.options['sample_size']
        else:
            fator_escala = 1.0

        daep_dx_full, daep_dy_full = wfm.aep_gradients(**grad_kwargs)

        # Em vez de sum(), use a média ou um fator de escala que não seja a soma total da grade
        # Se você somar 23 velocidades + direções, o gradiente explode.
        daep_dx = np.mean(daep_dx_full, axis=0) # Média preserva a ordem de grandeza
        daep_dy = np.mean(daep_dy_full, axis=0) 

        # E multiplique por um fator de ajuste (o 1e6 deles) 
        # para trazer o gradiente para perto da escala das coordenadas (centenas/milhares de metros)
        J['aep', 'x'] = (daep_dx * 1e6).reshape(1, -1)


# =============================================================================
# METODOLOGIA DE TESTE E VALIDAÇÃO DE IMPACTO DE VIZINHOS
# =============================================================================

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
    stochastic_mode=False, 
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
    stochastic_mode=True,  
    sample_size=50,        
    x_offset=x_offset,
    y_offset=y_offset,
    neighbor_x=neighbor_x, 
    neighbor_y=neighbor_y  
)
prob_stoch.model.add_subsystem('aep_stoch', comp_stoch, promotes=['*'])
prob_stoch.setup()
prob_stoch.set_val('x', x_real_active)
prob_stoch.set_val('y', y_real_active)

print("Avaliando convergência da escala estocástica por amostragem...")
for i in range(1, 10):
    prob_stoch.run_model()
    print(f"   Amostra SGD Iter {i} - AEP Escalada Estimada: {prob_stoch.get_val('aep')[0]:,.2f} MWh")

# print("\n=== 4. VALIDANDO OS GRADIENTES DO ESTOCÁSTICO ===")
# # Desliga a amostragem contínua de vento apenas durante o check_partials para o FD bater
# prob_stoch.model.aep_stoch.options['stochastic_mode'] = False
# prob_stoch.check_partials(compact_print=True, method='fd', step=1e-3)

# print("\n=== 5. AUTHENTICATING REAL NEIGHBORING IMPACT ===")
# prob_isolated = om.Problem()
# comp_no_neighbor = AEPComp(
#     n_wt=N_WT,
#     system_data=system_dat,
#     site_data=site_resource_dict,  
#     stochastic_mode=False, # Modo determinístico estável para comparação direta             
#     sample_size=1,                  
#     x_offset=x_offset,
#     y_offset=y_offset,
#     neighbor_x=np.array([]),
#     neighbor_y=np.array([])
# )
# prob_isolated.model.add_subsystem('aep_no_neighbor', comp_no_neighbor, promotes=['*'])
# prob_isolated.setup()
# prob_isolated.set_val('x', x_real_active)
# prob_isolated.set_val('y', y_real_active)
# prob_isolated.run_model()

# aep_no_neighbor = prob_isolated.get_val('aep')[0]
# loss_mwh = aep_no_neighbor - aep_det_real
# loss_percentage = (loss_mwh / aep_no_neighbor) * 100

# print(f"AEP Gaussiana com planta vizinha LIGADA:  {aep_det_real:,.2f} MWh")
# print(f"AEP Gaussiana com planta vizinha DESLIGADA: {aep_no_neighbor:,.2f} MWh")
# print(f"--> [Resultado] Planta vizinha rouba: {loss_mwh:,.2f} MWh da planta ativa.")
# print(f"--> [Impacto Real]: {loss_percentage:.2f}% de perdas por esteiras externas confirmadas.")

# if loss_mwh > 1e-3:
#     print("[AUTHENTICATED] Sucesso! O BastankhahGaussian acoplou perfeitamente os efeitos de fluxo do vizinho!")
# else:
#     print("[ALERT] O impacto deu zero. Verifique as posições relativas dos parques.")