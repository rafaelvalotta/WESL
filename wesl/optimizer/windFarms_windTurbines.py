from py_wake import np
from py_wake.site._site import UniformWeibullSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.turbulence_models import STF2017TurbulenceModel, IECWeight
from py_wake.deficit_models.gaussian import BastankhahGaussian, ZongGaussian, TurboGaussianDeficit, NiayifarGaussian, BlondelSuperGaussianDeficit2023
from py_wake import NOJ, Fuga
from py_wake.wind_farm_models import PropagateDownwind
from py_wake.deficit_models import TurboNOJDeficit
from py_wake.superposition_models import LinearSum
from py_wake.tests.test_files import tfp 



# DEFICIT MODELS

def noj_WF_model(site, windTurbines):
    wf_model = NOJ(site, windTurbines)
    model_name = 'NOJ'
    return wf_model, model_name

def turboNoj_WF_model(site, windTurbines):
    wf_model = PropagateDownwind(site, windTurbines, wake_deficitModel=TurboNOJDeficit(use_effective_ws=True),
                                 superpositionModel=LinearSum(), turbulenceModel=STF2017TurbulenceModel()
                                 )
    return wf_model

def fuga_WF_model(site, windTurbines):
    luh_path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc'
    wf_model = Fuga(luh_path, site, windTurbines)
    return wf_model

# THE GAUSSIAN WAKE DEFICIT MODELS

def bastankhah_WF_model(site, windTurbines):
    wf_model = BastankhahGaussian(site, windTurbines, use_effective_ws=True)
    return wf_model

def zong_WF_model(site, windTurbines):
    wf_model = ZongGaussian(site, windTurbines, turbulenceModel=STF2017TurbulenceModel(), use_effective_ws=True)
    return wf_model

def niayifar_WF_model(site, windTurbines):
    wf_model = NiayifarGaussian(site, windTurbines, turbulenceModel=STF2017TurbulenceModel(), use_effective_ws=True)
    return wf_model
def turboGaussian_WF_model(site, windTurbines):
    wf_model = PropagateDownwind(site, windTurbines, wake_deficitModel=TurboGaussianDeficit(use_effective_ws=True),
                                 superpositionModel=LinearSum(), turbulenceModel=STF2017TurbulenceModel()
                                 )
    return wf_model

def blondelSuperGaussian_WF_model(site, windTurbines):
    wf_model = PropagateDownwind(site, windTurbines, wake_deficitModel=BlondelSuperGaussianDeficit2023(use_effective_ws=True),
                                 superpositionModel=LinearSum(), turbulenceModel=STF2017TurbulenceModel()
                                 )
    return wf_model

## WIND TURBINES MODELS

class AD_5_116(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        hub height site specific
        """
        GenericWindTurbine.__init__(self, name='AD-5-116', diameter=116, hub_height=90,
                             power_norm=5000, turbulence_intensity=0.07)

class B32_450(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        hub height taken from B37/450
        """
        GenericWindTurbine.__init__(self, name='B32/450', diameter=32, hub_height=40,
                             power_norm=450, turbulence_intensity=0.07)

class B82_2300(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='B82/2300', diameter=82.4, hub_height=90,
                             power_norm=2300, turbulence_intensity=0.07)

class H150_6(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='H150_6', diameter=150, hub_height=100,
                             power_norm=6000, turbulence_intensity=0.07)

class Haliade_X(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='Haliade-X', diameter=220, hub_height=150,
                             power_norm=13000, turbulence_intensity=0.07)

class Haliade_X_13(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='Haliade-X 13', diameter=220, hub_height=150,
                             power_norm=13000, turbulence_intensity=0.07)
        
class Haliade_X_14(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='Haliade-X 14', diameter=220, hub_height=150,
                             power_norm=14000, turbulence_intensity=0.07)


 
class HC_Moura(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='HC_Moura', diameter=240, hub_height=180,
                             power_norm=15000, turbulence_intensity=0.07)

class MySE_185(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        hub height not provided
        """
        GenericWindTurbine.__init__(self, name='MySE 18.5', diameter=260, hub_height=180,
                             power_norm=18500, turbulence_intensity=0.12)

class SWT_23_82(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='SWT-23-82', diameter=82.4, hub_height=100,
                             power_norm=2300, turbulence_intensity=0.07)  
        
class SWT_23_93(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='SWT-23-93', diameter=90, hub_height=100,
                             power_norm=2300, turbulence_intensity=0.07)        

class SWT_36_107(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='SWT-36-107', diameter=107, hub_height=80,
                             power_norm=3600, turbulence_intensity=0.07) 

class SWT_36_120(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='SWT-36-120', diameter=120, hub_height=90,
                             power_norm=3600, turbulence_intensity=0.07)              

class SWT_40_120(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='SWT-4.0-120', diameter=120, hub_height=90,
                             power_norm=4000, turbulence_intensity=0.07) 

class SWT_40_130(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='SWT-4.0-130', diameter=130, hub_height=90,
                             power_norm=4000, turbulence_intensity=0.07) 

class SWT_60_120(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='SWT-6.0-154', diameter=120, hub_height=84,
                             power_norm=6000, turbulence_intensity=0.07) 

class SWT_60_154(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='SWT-6.0-154', diameter=154, hub_height=110,
                             power_norm=6000, turbulence_intensity=0.07) 

class SWT_63_154(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='SWT-6.3-154', diameter=154, hub_height=120,
                             power_norm=6300, turbulence_intensity=0.07) 

class SWT_70_154(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='SWT-7.0-154', diameter=154, hub_height=120,
                             power_norm=4000, turbulence_intensity=0.07) 

class AREVA_M5000_116(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='AREVA M5000-116', diameter=116, hub_height=90,
                             power_norm=5000, turbulence_intensity=0.07) 
        
class Bard_50(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='Bard 5.0', diameter=122, hub_height=90,
                             power_norm=5000, turbulence_intensity=0.07) 

class V39_05(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='V39-0.5', diameter=39, hub_height=50,
                             power_norm=500, turbulence_intensity=0.07)

class V80_20(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='V80-2.0', diameter=80, hub_height=75,
                             power_norm=2000, turbulence_intensity=0.07)

class V90_30(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='V90-3.0', diameter=90, hub_height=85,
                             power_norm=3000, turbulence_intensity=0.07)

class V112_30(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='V112-3.0', diameter=112, hub_height=100,
                             power_norm=3000, turbulence_intensity=0.07)

class V112_33(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='V112-3.3', diameter=112, hub_height=85,
                             power_norm=3300, turbulence_intensity=0.07)

class MHI_V164_80(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='MHI V164-8.0', diameter=164, hub_height=105,
                             power_norm=8000, turbulence_intensity=0.07)

class MHI_V164_83(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='MHI V164-8.3', diameter=164, hub_height=100,
                             power_norm=8300, turbulence_intensity=0.07)   


class MHI_V164_84(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='MHI V164-8.4', diameter=164, hub_height=100,
                             power_norm=8400, turbulence_intensity=0.07)        
        #diameter not found 11/21/24, substituted values from V164 8.0

class MHI_V164_95(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='MHI V164-9.5', diameter=164, hub_height=105,
                             power_norm=9500, turbulence_intensity=0.07)

class V236_150(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='V236-15.0', diameter=236, hub_height=180,
                             power_norm=15000, turbulence_intensity=0.07)        
        #diameter not found 11/21/24, substituted values from V164 8.0

class Senvion_50M126(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        power_norm taken from 6.2M126
        """
        GenericWindTurbine.__init__(self, name='Senvion 5.0M126', diameter=126, hub_height=90,
                             power_norm=5000, turbulence_intensity=0.07)

class Senvion_62M126(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='Senvion 6.2M126', diameter=126, hub_height=90,
                             power_norm=6150, turbulence_intensity=0.07)

class Senvion_62M152(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='Senvion 6.2M152', diameter=152, hub_height=122,
                             power_norm=6150, turbulence_intensity=0.07)

class SG_80_167_DD(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='SG 8.0 167 DD', diameter=167, hub_height=119,
                             power_norm=8000, turbulence_intensity=0.07)
        
class SG_110_200_DD(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='SG 11.0-200 DD', diameter=200, hub_height=140,
                             power_norm=11000, turbulence_intensity=0.08)
       
class SG_140_222_DD(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='SG-140-222 DD', diameter=222, hub_height=150,
                             power_norm=14000, turbulence_intensity=0.07)

class SG_140_236_DD(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='SG-140-236 DD', diameter=236, hub_height=150,
                             power_norm=14000, turbulence_intensity=0.07)

class B32_450(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='B32-450', diameter=35, hub_height=35,
                             power_norm=450, turbulence_intensity=0.07)

class N900_2300(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        """
        GenericWindTurbine.__init__(self, name='N900/2300', diameter=90, hub_height=80,
                             power_norm=2300, turbulence_intensity=0.07)

class E70_2300(GenericWindTurbine):
    def __init__(self):
        """
        Parameters
        ----------
        The turbulence intensity Varies around 6-8%
        Hub Height Site Specific
        """
        GenericWindTurbine.__init__(self, name='E70/2300', diameter=71, hub_height=85,
                             power_norm=2300, turbulence_intensity=0.07)

## WIND FARMS MODELS - US EAST COAST

# Revolution Wind and Shouth Fork Wind Turibnes Postions
RevolutionSouthForkWT_x= [317826.0105875855, 319708.9459259393, 321591.8539341013, 323428.88512329943, 325221.05184876826, 
                          327148.71417500044, 327151.1722056244, 325267.87560820964, 323384.5491548496, 321547.05686860136, 
                          317917.9386384081, 315987.48691129254, 314337.50893202034, 315988.3253023663, 317918.1897189942, 
                          319753.9341844553, 321502.26103143406, 323383.8712021542, 323432.51548316237, 321503.4402310264, 
                          319663.866344754, 317872.49882155954, 315943.3041763123, 314195.5691159735, 310377.84630411875, 
                          312306.42158869, 314145.4640580319, 316168.2395666874, 317824.529144102, 319708.2473693589, 
                          321548.3008626302, 323433.097063238, 325270.87269411265, 327156.7067862863, 328995.5238493092, 
                          330880.21175658563, 332717.8435975673, 334510.6144870524, 325274.0596461869, 329089.3755504235, 
                          325277.356208049, 327209.0326196457, 329137.3840026625, 327118.6343406278, 329002.62393320084, 
                          327072.0783478966, 329052.82498086244, 327166.80507091375, 328915.8840048452, 330713.09796848346, 
                          332736.06149291823, 334485.1495123971, 336510.17426589003, 336324.22431479226, 334389.6310686593, 
                          332691.206576428, 330892.4116338047, 332680.8949319166, 338157.24725721474, 340044.2419439816, 
                          341930.1949565129, 343861.1553572938, 345654.1629410349, 345556.9334575982, 343716.44241562975, 
                          341923.881130798, 340085.31703607907, 321548.5908471775, 319710.5926481589, 317824.3140298442, 
                          315983.9404751969, 315986.1762309221, 317823.4814913715, 321643.0033623148, 323484.8710893603, 
                          323438.45762136386, 321551.2659058167, 319713.38918979047, 317872.1333274647, 315936.63764927746]

RevolutionSouthForkWT_y= [4566088.0271595, 4566086.94508547, 4566086.349543752, 4566087.351905313, 4566135.550198147, 
                          4566089.546583842, 4564259.3493943205, 4564258.604964562, 4564258.235963927, 4564257.230274632, 
                          4564255.45999023, 4564212.405289804, 4562560.697743491, 4562427.309387789, 4562424.6357088685, 
                          4562333.258323682, 4562427.560415846, 4562335.97503257, 4560549.498378367, 4560596.437273307, 
                          4560550.023600059, 4560594.51963829, 4560642.931506823, 4560595.687392474, 4558724.672449198, 
                          4558629.063954246, 4558627.788301046, 4558622.346310797, 4558672.480721551, 4558625.718799355, 
                          4558672.048056355, 4558672.099578482, 4558627.883463959, 4558674.5260739885, 4558676.988628905, 
                          4558678.961759021, 4558636.622624232, 4558641.527562989, 4556841.545291079, 4556797.047537556, 
                          4555054.765822244, 4555054.7209210005, 4554917.74359634, 4553178.099980796, 4553088.0399587955, 
                          4551208.359593137, 4551299.339017954, 4549372.3648296, 4549377.052868125, 4549472.907844638, 
                          4549380.538557856, 4549386.665284371, 4549386.931248458, 4551362.232803121, 4551268.24942627, 
                          4551444.401307117, 4551256.594875634, 4553002.961696985, 4553108.870961092, 4553158.989745676, 
                          4553163.743262109, 4553122.260223048, 4553084.197666842, 4555010.413976396, 4554958.016500382, 
                          4554996.567193426, 4555036.564251611, 4553036.954329632, 4553127.912643072, 4553128.77562025, 
                          4553128.96901014, 4551387.088020707, 4551249.466135039, 4551247.0244748285, 4551294.010300911, 
                          4549369.5645569675, 4549369.43592957, 4549506.217712882, 4549505.906598289, 4549416.77488112]

# Dummy_wind_turbine initial positions:
dummyWT_x = [100000, 100250, 100500, 100750, 101000, 101250, 101500, 101750, 102000,
             100000, 100250, 100500, 100750, 101000, 101250, 101500, 101750, 102000,
             100000, 100250, 100500, 100750, 101000, 101250, 101500, 101750, 102000,
             100000, 100250, 100500, 100750, 101000, 101250, 101500, 101750, 102000,
             ]
dummyWT_y = [4510000, 4510000, 4510000, 4510000, 4510000, 4510000, 4510000, 4510000, 4510000,
             4515000, 4515000, 4515000, 4515000, 4515000, 4515000, 4515000, 4515000, 4515000,
             4520000, 4520000, 4520000, 4520000, 4520000, 4520000, 4520000, 4520000, 4520000,
             4525000, 4525000, 4525000, 4525000, 4525000, 4525000, 4525000, 4525000, 4525000,
             ]


# Vineyard Wind Turbine Positions
VineyardWT_x = [377155.2519440331, 378998.7970951452, 377157.34397726814, 375317.5471674723, 373410.2996410408, 
                375219.4106331823, 377094.72025408107, 379001.3699542344, 371568.48875894566, 373410.92229268537, 
                375253.8948291801, 377161.5001290994, 379069.6455727225, 380781.0014982596, 382654.1556128802, 
                384502.55561329535, 382757.88922419725, 380848.77358589135, 378971.9285817933, 377130.7159173434, 
                375287.29973818973, 373444.4157358758, 371634.9374684697, 369660.45196643466, 367849.61256236467, 
                369725.96101338125, 371569.3447769246, 373445.6524201946, 375321.2995053398, 377165.1846616655, 
                379008.49187318474, 380884.72234694136, 382728.5173007832, 384571.2435932317, 384541.8187336057, 
                382731.572749875, 380822.02538487973, 378945.3231056019, 377068.0524851483, 375192.9788065369, 
                373381.5715055847, 371537.753488039, 369660.3808966652, 367882.3357515128, 365972.6323078332, 
                367849.2589278419, 369758.83920039545, 371570.8511897628, 373381.6981324416, 375390.5217229137, 
                377168.94409278454, 379046.1431337041, 380857.4873542791, 382701.6894058671, 380827.1356001375, 
                379015.33003719104, 377137.6528697772, 375228.1893457855, 373448.74452166556, 375294.7004336106, 
                377172.8663313822, 378984.52012715384, 377108.61205533706
            ]

VineyardWT_y = [4554942.059508868, 4553109.896111671, 4553107.8401351785, 4553204.497728261, 4551304.328024954, 
                4551306.010780756, 4551339.787196802, 4551275.248479973, 4549435.677996828, 4549436.348097042, 
                4549470.231551096, 4549437.975089849, 4549439.0766765615, 4549476.524509717, 4549347.874762597, 
                4547646.949875426, 4547609.03997222, 4547574.328722003, 4547506.7110762615, 4547602.991380055, 
                4547568.514506935, 4547567.361821613, 4547598.846713559, 4547600.952698807, 4545731.621433114, 
                4545730.903292232, 4545731.1314991545, 4545731.254305482, 4545699.109093857, 4545733.604890574, 
                4545735.70348551, 4545737.730081297, 4545773.624304078, 4545744.372004114, 4543875.659268664, 
                4543904.382198243, 4543902.40712253, 4543900.275523934, 4543865.874540068, 4543963.17381269, 
                4543928.594985781, 4543927.900842147, 4543895.394985081, 4543894.449959907, 4543896.340396929, 
                4542058.025412571, 4542023.885222569, 4542057.468234255, 4542025.999173151, 4542024.482636555, 
                4542027.257484455, 4542028.9451462375, 4542032.07046016, 4542035.228189824, 4540162.364155633, 
                4540159.236468282, 4540157.546304386, 4540222.615673676, 4540187.440831584, 4538317.924037515, 
                4538319.153527374, 4538289.08887008, 4536449.004810518]





class VineyardWind1(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.4633, 7.6414, 6.3740, 5.9969, 4.7711, 4.5698, 
            7.3598, 11.8051, 13.2464, 11.0975, 11.1503, 9.5244]
        a = [10.19, 10.45, 9.47, 9.02, 9.48, 9.66, 
            11.44, 13.27, 12.46, 11.36, 12.39, 10.45]
        k = [2.170, 1.725, 1.713, 1.682, 1.521, 1.479,
            1.666, 2.143, 2.385, 2.146, 2.432, 2.373]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([VineyardWT_x, VineyardWT_y]).T
        self.name = "Vineyard Wind Farm"


ClusterWT_x = [
    322000, 327000, 332000, 337000, 342000, 347000, 352000, 357000, 362000, 367000, 372000, 377000, 382000, 387000, 392000,
    325000, 330000, 335000, 340000, 345000, 350000, 355000, 360000, 365000, 370000, 375000, 380000, 385000, 390000,
    322000, 327000, 332000, 337000, 342000, 347000, 352000, 357000, 362000, 367000, 372000, 377000, 382000, 387000, 392000,
    325000, 330000, 335000, 340000, 345000, 350000, 355000, 360000, 365000, 370000, 375000, 380000, 385000, 390000,
    322000, 327000, 332000, 337000, 342000, 347000, 352000, 357000, 362000, 367000, 372000, 377000, 382000, 387000, 392000,
    325000, 330000, 335000, 340000, 345000, 350000, 355000, 360000, 365000, 370000, 375000, 380000, 385000, 390000,
    322000, 327000, 332000, 337000, 342000, 347000, 352000, 357000, 362000, 367000, 372000, 377000, 382000, 387000, 392000,
    325000, 330000, 335000, 340000, 345000, 350000, 355000, 360000, 365000, 370000, 375000, 380000, 385000, 390000,
    322000, 327000, 332000, 337000, 342000, 347000, 352000, 357000, 362000, 367000, 372000, 377000, 382000, 387000, 392000
]


ClusterWT_y = [
    # Each row repeats a Y-value, corresponding to the respective staggered X-rows above
    4510000, 4510000, 4510000, 4510000, 4510000, 4510000, 4510000, 4510000, 4510000, 4510000, 4510000, 4510000, 4510000, 4510000, 4510000,
    4515000, 4515000, 4515000, 4515000, 4515000, 4515000, 4515000, 4515000, 4515000, 4515000, 4515000, 4515000, 4515000, 4515000,
    4520000, 4520000, 4520000, 4520000, 4520000, 4520000, 4520000, 4520000, 4520000, 4520000, 4520000, 4520000, 4520000, 4520000, 4520000,
    4525000, 4525000, 4525000, 4525000, 4525000, 4525000, 4525000, 4525000, 4525000, 4525000, 4525000, 4525000, 4525000, 4525000,
    4530000, 4530000, 4530000, 4530000, 4530000, 4530000, 4530000, 4530000, 4530000, 4530000, 4530000, 4530000, 4530000, 4530000, 4530000,
    4535000, 4535000, 4535000, 4535000, 4535000, 4535000, 4535000, 4535000, 4535000, 4535000, 4535000, 4535000, 4535000, 4535000,
    4540000, 4540000, 4540000, 4540000, 4540000, 4540000, 4540000, 4540000, 4540000, 4540000, 4540000, 4540000, 4540000, 4540000, 4540000,
    4545000, 4545000, 4545000, 4545000, 4545000, 4545000, 4545000, 4545000, 4545000, 4545000, 4545000, 4545000, 4545000, 4545000,
    4550000, 4550000, 4550000, 4550000, 4550000, 4550000, 4550000, 4550000, 4550000, 4550000, 4550000, 4550000, 4550000, 4550000, 4550000
]




class clusterWF_EastUS(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.4633, 7.6414, 6.3740, 5.9969, 4.7711, 4.5698, 
            7.3598, 11.8051, 13.2464, 11.0975, 11.1503, 9.5244]
        a = [10.19, 10.45, 9.47, 9.02, 9.48, 9.66, 
            11.44, 13.27, 12.46, 11.36, 12.39, 10.45]
        k = [2.170, 1.725, 1.713, 1.682, 1.521, 1.479,
            1.666, 2.143, 2.385, 2.146, 2.432, 2.373]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([ClusterWT_x, ClusterWT_y]).T
        self.name = "Cluster WF East US"


SunriseWT_x = [
    393000, 394000, 395000, 396000, 397000, 398000, 399000, 400000, 401000, 402000,  # Row 1
    393500, 394500, 395500, 396500, 397500, 398500, 399500, 400500, 401500, 402500,  # Row 2
    393000, 394000, 395000, 396000, 397000, 398000, 399000, 400000, 401000, 402000,  # Row 3
    393500, 394500, 395500, 396500, 397500, 398500, 399500, 400500, 401500, 402500,  # Row 4
    393000, 394000, 395000, 396000, 397000, 398000, 399000, 400000, 401000, 402000,  # Row 5
    393500, 394500, 395500, 396500, 397500, 398500, 399500, 400500, 401500, 402500,  # Row 6
    393000, 394000, 395000, 396000, 397000, 398000, 399000, 400000, 401000, 402000,  # Row 7
    393500, 394500, 395500, 396500, 397500, 398500, 399500, 400500, 401500, 402500,  # Row 8
    393000, 394000, 395000, 396000, 397000, 398000, 399000, 400000, 401000, 402000   # Row 9
]

SunriseWT_y = [
    4532000, 4532000, 4532000, 4532000, 4532000, 4532000, 4532000, 4532000, 4532000, 4532000,  # Row 1
    4532800, 4532800, 4532800, 4532800, 4532800, 4532800, 4532800, 4532800, 4532800, 4532800,  # Row 2
    4533600, 4533600, 4533600, 4533600, 4533600, 4533600, 4533600, 4533600, 4533600, 4533600,  # Row 3
    4534400, 4534400, 4534400, 4534400, 4534400, 4534400, 4534400, 4534400, 4534400, 4534400,  # Row 4
    4535200, 4535200, 4535200, 4535200, 4535200, 4535200, 4535200, 4535200, 4535200, 4535200,  # Row 5
    4536000, 4536000, 4536000, 4536000, 4536000, 4536000, 4536000, 4536000, 4536000, 4536000,  # Row 6
    4536800, 4536800, 4536800, 4536800, 4536800, 4536800, 4536800, 4536800, 4536800, 4536800,  # Row 7
    4537600, 4537600, 4537600, 4537600, 4537600, 4537600, 4537600, 4537600, 4537600, 4537600,  # Row 8
    4538400, 4538400, 4538400, 4538400, 4538400, 4538400, 4538400, 4538400, 4538400, 4538400   # Row 9
]


class sunrise_wind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.4633, 7.6414, 6.3740, 5.9969, 4.7711, 4.5698, 
            7.3598, 11.3051, 12.2464, 11.0975, 11.1503, 10.2244]
        a = [10.19, 10.45, 9.47, 9.02, 9.48, 9.66, 
            11.44, 13.27, 14.46, 12.36, 12.39, 10.45]
        k = [2.170, 1.725, 1.713, 1.682, 1.521, 1.479,
            1.676, 2.143, 3.085, 2.646, 2.432, 2.373]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([SunriseWT_x, SunriseWT_y]).T
        self.name = "Sunrise Wind"

class Bluepoint_wind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [8.2195, 6.0248, 7.8295, 5.9751, 4.6214, 4.3304, 
             7.1784, 13.3585, 11.8165, 9.5303, 11.7679, 9.3479]
        
        a = [9.96, 10.02, 10.27, 9.08, 8.49, 8.2, 10.36, 12.96, 
             12.25, 11.21, 12.39, 10.96]
        
        k = [2.178, 1.854, 1.807, 1.658, 1.463, 1.42, 1.561, 1.893, 
             1.994, 2.045, 2.314, 2.385]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Bluepoint Wind"


class California_Long_Bay(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [7.407, 10.1427, 10.008, 7.0119, 4.8164, 6.1157, 8.5767, 15.4914, 14.5195, 5.8756, 4.562, 5.473]

        a = [10.76, 10.56, 9.39, 7.47, 6.49, 7.0, 8.34, 10.21, 10.71, 9.36, 9.89, 10.18]

        k = [2.314, 2.229, 2.26, 1.768, 1.689, 1.607, 1.439, 1.936, 2.326, 1.76, 2.033, 2.303]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "California Long Bay"

class Coastal_Virginia_Wind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [9.1344, 9.8471, 8.875, 5.3448, 4.5592, 5.3863, 11.1043, 14.9041, 9.3235, 5.0583, 6.896, 9.567]
        a = [10.37, 9.93, 9.18, 8.13, 7.63, 7.93, 11.04, 13.5, 11.88, 9.75, 10.12, 11.31]
        k = [2.225, 2.143, 2.014, 1.744, 1.588, 1.467, 1.842, 2.447, 2.682, 1.881, 1.916, 2.326]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Coastal Virginia Wind"

class Community_offshore_wind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [8.1634, 6.243, 7.9995, 5.9399, 4.9283, 4.4364, 7.3568, 13.9537, 11.4867, 8.0738, 11.9819, 9.4366]
        a = [10.24, 9.62, 10.2, 9.03, 8.89, 8.43, 10.25, 13.01, 12.34, 10.96, 12.65, 10.85]
        k = [2.283, 1.854, 1.811, 1.701, 1.549, 1.447, 1.529, 1.881, 2.041, 1.963, 2.475, 2.424]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Community Offshore Wind"

class Kittyhawk(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [9.6309, 10.7616, 8.9617, 5.4004, 4.0631, 4.6647, 8.8588, 16.5171, 10.6597, 5.4015, 5.9926, 9.088]
        a = [10.64, 9.98, 9.12, 8.41, 7.62, 7.63, 9.83, 13.48, 12.03, 10.41, 10.15, 11.51]
        k = [2.15, 1.963, 1.963, 1.865, 1.631, 1.408, 1.604, 2.104, 2.311, 2.029, 1.912, 2.342]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Kittyhawk"

class Leading_Light_wind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [8.2848, 6.1951, 8.3466, 5.7296, 5.0847, 4.3328, 7.392, 14.351, 11.2171, 7.9369, 11.7598, 9.3696]
        a = [10.31, 9.68, 10.04, 8.86, 8.9, 8.53, 10.4, 13.19, 12.34, 11.07, 12.7, 10.83]
        k = [2.279, 1.881, 1.771, 1.654, 1.553, 1.447, 1.619, 1.912, 2.072, 2.025, 2.498, 2.439]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Leading Light Wind"

class OceanWind_2A(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [7.8069, 5.2563, 7.848, 6.0721, 5.3821, 4.7507, 8.7824, 15.094, 10.0293, 7.7138, 12.1654, 9.0991]
        a = [10.38, 9.67, 10.16, 8.21, 8.09, 7.96, 10.15, 13.0, 11.61, 10.71, 12.61, 11.44]
        k = [2.877, 1.732, 1.975, 1.475, 1.514, 1.424, 1.635, 1.924, 2.158, 2.006, 2.596, 2.869]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "OceanWind 2A"

class OceanWind_2B(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [7.5913, 5.6912, 7.8906, 5.8392, 5.0406, 4.7135, 8.4621, 15.7622, 9.5643, 8.1995, 12.2015, 9.0439]
        a = [10.28, 9.83, 9.95, 8.27, 8.22, 7.95, 10.47, 13.47, 11.85, 10.29, 12.73, 11.13]
        k = [2.65, 1.799, 1.928, 1.525, 1.568, 1.428, 1.654, 2.006, 2.146, 1.857, 2.619, 2.717]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "OceanWind 2B"

class SkipJack_wind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [7.1912, 6.2479, 7.7666, 5.1501, 4.3038, 4.8409, 9.3739, 17.2451, 8.7507, 6.4581, 12.3837, 10.2881]
        a = [10.01, 9.86, 9.66, 8.27, 7.69, 8.52, 10.62, 13.69, 11.64, 9.85, 12.01, 10.77]
        k = [2.479, 1.908, 2.021, 1.725, 1.564, 1.553, 1.666, 2.283, 2.615, 1.779, 2.346, 2.369]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Skip Jack Wind"

class US_wind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [8.199, 6.9222, 7.7897, 5.2174, 4.3906, 4.8008, 9.7826, 17.522, 8.7398, 5.6747, 11.2841, 9.6771]
        a = [9.83, 9.7, 9.67, 8.22, 8.01, 8.75, 10.55, 13.97, 12.65, 10.14, 12.83, 11.32]
        k = [2.236, 1.869, 2.018, 1.682, 1.525, 1.561, 1.533, 2.033, 2.432, 1.818, 2.678, 2.572]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "US Wind"


# European Wind Farm Classes
        
class Alpha_Ventus(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.052, 3.9934, 5.8218, 8.3642, 7.4577, 6.0229, 7.5221, 12.1841, 14.3322, 12.3524, 9.745, 8.1523]
        a = [8.13, 7.91, 9.45, 10.87, 11.1, 10.5, 11.13, 13.1, 12.45, 11.91, 10.98, 9.78]
        k = [2.037, 2.213, 2.201, 2.248, 2.822, 2.689, 2.135, 2.693, 2.381, 2.33, 2.451, 2.115]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Alpha Ventus"        

class Amrumbank_west(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.3608, 3.6481, 5.764, 8.2902, 7.4711, 6.1776, 6.8845, 11.6478, 14.0379, 12.5572, 11.2562, 7.9045] 
        a = [8.34, 7.53, 9.85, 11.32, 11.09, 10.79, 11.06, 12.93, 12.57, 11.72, 11.26, 10.0] 
        k = [2.283, 2.275, 2.471, 2.627, 2.678, 2.564, 2.158, 2.678, 2.389, 2.326, 2.537, 2.076]         
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Amrumbank West"

class Anholt(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.2114, 3.8168, 4.9063, 6.6518, 7.5065, 7.8457, 9.3587, 13.6546, 12.0781, 16.3373, 9.379, 4.2537]
        a = [8.16, 7.93, 8.71, 10.15, 11.31, 10.61, 9.81, 12.56, 12.18, 12.11, 11.06, 8.37]
        k = [1.967, 2.131, 2.236, 2.24, 2.494, 2.307, 2.119, 3.119, 2.967, 2.604, 2.178, 1.783]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Anholt"

class Anholt_2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [3.8331, 3.9226, 5.4095, 6.2996, 8.0452, 8.6782, 8.7888, 11.9816, 13.7982, 16.1181, 9.3434, 3.7816]
        a = [8.97, 8.36, 9.06, 10.4, 11.72, 11.25, 11.47, 13.14, 13.26, 13.65, 12.5, 9.54]
        k = [2.162, 1.971, 2.061, 2.053, 2.291, 2.342, 2.205, 2.525, 2.408, 2.748, 2.182, 1.822]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Anholt 2"

class Bard_offshore_1(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.4532, 4.0086, 5.8195, 7.5597, 7.4223, 5.9482, 7.6993, 12.2797, 14.5779, 12.1087, 9.5303, 8.5924]
        a = [8.65, 7.94, 8.87, 11.13, 11.32, 10.92, 11.18, 12.71, 12.56, 11.96, 11.33, 10.02]
        k = [2.17, 2.252, 2.037, 2.322, 2.596, 2.627, 1.951, 2.443, 2.393, 2.303, 2.498, 2.127]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Bard Offshore 1"

class Beatric_and_moray_east_west(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.0327, 3.895, 3.0428, 4.8718, 9.6479, 12.2913, 8.7185, 9.5472, 16.2751, 10.9662, 6.8954, 7.8163] 
        a = [10.91, 7.9, 5.8, 8.14, 10.78, 12.81, 11.54, 11.25, 13.39, 12.72, 10.95, 10.94] 
        k = [2.252, 1.701, 1.354, 1.826, 2.131, 2.807, 2.467, 2.162, 2.529, 2.209, 1.955, 2.201] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Beatric and Moray East West"

class Blyth_demo_phase1(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.3704, 4.8756, 4.9487, 4.5646, 6.9253, 8.2443, 6.5851, 9.91, 11.8098, 19.5343, 8.5085, 7.7233] 
        a = [9.48, 7.59, 8.52, 7.97, 8.85, 10.28, 11.93, 12.68, 13.83, 15.42, 13.55, 12.36] 
        k = [1.998, 1.646, 1.936, 1.568, 1.787, 2.096, 2.225, 2.197, 2.029, 2.424, 2.842, 2.775] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Blyth Demo Phase1"

class Borkum_Riffgrund_1(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.0448, 3.9953, 5.837, 8.3644, 7.4644, 5.9997, 7.5451, 12.1703, 14.3213, 12.3438, 9.7875, 8.1264]
        a = [8.13, 7.91, 9.46, 10.88, 11.11, 10.5, 11.14, 13.08, 12.46, 11.9, 10.94, 9.8]
        k = [2.041, 2.217, 2.209, 2.256, 2.83, 2.697, 2.15, 2.693, 2.381, 2.33, 2.432, 2.119]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Borkum Riffgrund 1"

class Borkum_Riffgrund_2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [3.9985, 3.9629, 5.7423, 8.303, 7.4497, 6.0488, 7.4897, 12.3506, 14.4247, 12.4075, 9.6396, 8.1826]
        a = [8.22, 7.98, 9.33, 10.8, 11.06, 10.44, 11.12, 13.08, 12.51, 11.89, 11.02, 9.67]
        k = [2.064, 2.279, 2.17, 2.221, 2.943, 2.67, 2.123, 2.674, 2.404, 2.299, 2.479, 2.1]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Borkum Riffgrund 2"

class Borkum_Riffgrund_3(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.1294, 4.0013, 5.7769, 8.0761, 7.5983, 5.932, 7.5791, 12.1597, 14.621, 12.223, 9.6104, 8.2927]
        a = [8.58, 7.98, 9.36, 11.05, 11.3, 10.82, 11.65, 13.64, 12.96, 12.45, 11.37, 9.94]
        k = [2.088, 2.107, 2.033, 2.045, 2.584, 2.443, 1.982, 2.521, 2.295, 2.311, 2.365, 2.068]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Borkum Riffgrund 3"

class Borkum_Riffgrund_4(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.052, 3.9934, 5.8218, 8.3642, 7.4577, 6.0229, 7.5221, 12.1841, 14.3322, 12.3524, 9.745, 8.1523]
        a = [8.13, 7.91, 9.45, 10.87, 11.1, 10.5, 11.13, 13.1, 12.45, 11.91, 10.98, 9.78]
        k = [2.037, 2.213, 2.201, 2.248, 2.822, 2.689, 2.135, 2.693, 2.381, 2.33, 2.451, 2.115]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Borkum Riffgrund 4"

class Borssele_1(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.6383, 6.8467, 6.7623, 7.0617, 5.5745, 4.7049, 7.6737, 12.8036, 18.9796, 9.8714, 7.409, 6.6743]
        a = [8.89, 8.96, 9.56, 9.76, 8.61, 8.64, 11.79, 12.45, 12.32, 10.82, 9.38, 9.36]
        k = [2.342, 2.467, 2.623, 2.26, 2.369, 1.943, 2.346, 2.436, 2.326, 2.236, 2.014, 2.064]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Borssele 1"

class Borssele_2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.6228, 7.1077, 6.9181, 7.1251, 5.4138, 4.5289, 7.9069, 12.3957, 18.8971, 10.0692, 7.4815, 6.5333]
        a = [8.81, 8.98, 9.51, 10.02, 8.54, 8.72, 11.85, 12.34, 12.49, 10.43, 9.25, 9.22]
        k = [2.322, 2.459, 2.654, 2.576, 2.459, 1.982, 2.404, 2.518, 2.389, 2.143, 1.986, 2.045]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Borssele 2"        

class Borssele_3(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.6897, 6.7066, 6.8727, 7.1117, 5.5248, 4.6787, 7.8552, 12.9489, 19.0572, 9.709, 7.4003, 6.4451]
        a = [9.06, 8.92, 9.5, 9.88, 8.55, 8.59, 11.61, 12.61, 12.3, 10.8, 9.45, 9.27]
        k = [2.373, 2.49, 2.568, 2.318, 2.451, 1.908, 2.256, 2.451, 2.314, 2.244, 1.99, 2.014]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Borssele 3"        

class Borssele_4(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.5941, 6.9431, 6.9249, 7.1138, 5.4848, 4.5117, 7.9913, 12.3849, 19.2422, 9.9205, 7.3173, 6.5715]
        a = [8.98, 8.91, 9.52, 9.83, 8.56, 8.69, 11.72, 12.45, 12.43, 10.61, 9.46, 9.18]
        k = [2.377, 2.443, 2.643, 2.338, 2.459, 1.955, 2.338, 2.467, 2.361, 2.174, 2.037, 2.002]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Borssele 4"               

class Borssele_5(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.6494, 6.8413, 6.8281, 7.1198, 5.527, 4.6606, 7.8087, 12.7249, 19.0201, 9.8741, 7.348, 6.5978]
        a = [8.97, 8.92, 9.54, 9.84, 8.54, 8.62, 11.76, 12.49, 12.35, 10.98, 9.44, 9.32]
        k = [2.373, 2.451, 2.623, 2.318, 2.396, 1.936, 2.322, 2.443, 2.334, 2.404, 2.025, 2.045]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Borssele 5"                 

class Butendiek(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.2245, 3.3479, 5.2887, 8.3151, 7.3607, 6.2491, 6.8171, 11.7787, 13.8723, 12.6839, 11.8139, 8.2482]
        a = [8.28, 7.81, 9.71, 11.18, 11.33, 10.83, 11.12, 12.66, 12.49, 11.74, 11.75, 10.04]
        k = [2.174, 2.287, 2.369, 2.654, 2.666, 2.533, 2.178, 2.619, 2.354, 2.365, 2.67, 2.057]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Butendiek"        

class Dantysk(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.1941, 3.4725, 5.2155, 7.7413, 7.5294, 6.0959, 7.2592, 12.0755, 13.7112, 12.4711, 11.3082, 8.926]
        a = [8.21, 7.89, 9.54, 11.08, 11.43, 11.18, 11.37, 12.62, 12.6, 11.81, 11.84, 10.16]
        k = [1.967, 2.307, 2.283, 2.424, 2.521, 2.514, 2.182, 2.529, 2.346, 2.373, 2.682, 2.033]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Dantysk"

class Deutsche_Bucht(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.599, 4.072, 5.7497, 7.5935, 7.2634, 5.9875, 7.821, 12.4037, 14.5853, 12.0395, 9.342, 8.5434]
        a = [8.8, 7.86, 8.9, 11.0, 11.37, 10.91, 11.21, 12.6, 12.55, 12.08, 11.37, 10.02]
        k = [2.229, 2.232, 2.072, 2.283, 2.631, 2.611, 1.932, 2.396, 2.373, 2.369, 2.521, 2.139]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Deutsche Bucht"

class Dogger_BankA(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.1374, 4.332, 4.9315, 6.5943, 5.6886, 6.5531, 10.4321, 13.2211, 13.7967, 11.027, 8.9955, 8.2906]
        a = [9.57, 8.23, 9.05, 10.83, 10.59, 11.26, 12.2, 13.61, 13.77, 12.6, 11.06, 10.68]
        k = [2.15, 1.811, 2.338, 2.076, 2.08, 2.385, 1.826, 2.119, 2.494, 2.123, 2.166, 2.17]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Dogger BankA"

class Dogger_Bank_B(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.2431, 4.2708, 4.7881, 6.1057, 5.8307, 6.8715, 10.5863, 13.2743, 13.3976, 11.0512, 9.2523, 8.3283]
        a = [9.46, 8.26, 9.32, 11.04, 10.51, 11.29, 12.22, 13.66, 13.71, 12.81, 11.06, 10.42]
        k = [2.143, 1.846, 2.455, 2.146, 2.021, 2.377, 1.826, 2.131, 2.436, 2.189, 2.092, 1.959]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Dogger Bank B"        

class Dogger_Bank_C(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.6466, 4.2307, 5.0793, 6.7809, 5.9578, 6.5104, 9.8432, 12.6702, 14.0244, 10.844, 9.9855, 8.427]
        a = [9.66, 8.25, 8.87, 10.93, 11.35, 11.04, 12.09, 13.45, 13.32, 12.72, 11.2, 10.64]
        k = [2.146, 1.752, 2.104, 2.104, 2.346, 2.264, 1.826, 2.123, 2.291, 2.154, 2.143, 2.131]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Dogger Bank C"             

class Dogger_Bank_D_E(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.2123, 4.5152, 4.8123, 6.7873, 5.5661, 6.5338, 10.2852, 13.4578, 13.8072, 11.0135, 8.8949, 8.1144]
        a = [9.85, 8.18, 9.25, 10.92, 10.19, 11.46, 12.46, 14.31, 14.53, 12.96, 11.13, 10.46]
        k = [2.154, 1.768, 2.271, 1.994, 1.912, 2.162, 1.791, 2.131, 2.572, 2.162, 2.096, 1.936]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Dogger Bank D & E"      

class Dudgeon(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.5307, 4.2887, 5.2057, 6.6299, 5.612, 8.2647, 9.0571, 12.3369, 16.2943, 11.0778, 7.9457, 6.7565]
        a = [9.58, 8.71, 9.05, 9.88, 8.72, 10.53, 11.87, 13.24, 12.72, 11.81, 10.08, 9.71]
        k = [2.436, 2.404, 2.564, 2.232, 1.963, 2.588, 2.279, 2.615, 2.721, 2.365, 2.533, 2.025]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Dudgeon"  

class Dunkerque(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.0587, 7.0765, 7.6831, 6.4143, 3.8102, 4.7373, 6.7817, 14.7857, 19.9254, 10.9521, 6.9599, 5.815]
        a = [8.9, 8.77, 9.85, 10.12, 9.69, 9.32, 12.54, 14.87, 13.47, 10.25, 9.17, 9.13]
        k = [1.99, 2.217, 2.314, 2.232, 2.15, 1.775, 2.064, 3.045, 2.459, 1.893, 1.869, 1.814]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Dunkerque"

class East_Anglia_One(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.3104, 5.5518, 6.1207, 7.0576, 5.5497, 5.4949, 8.2572, 15.9459, 15.6538, 9.7558, 7.9609, 6.3413]
        a = [8.85, 8.79, 9.38, 9.78, 8.59, 8.75, 10.77, 13.27, 12.65, 11.21, 9.84, 9.81]
        k = [2.229, 2.479, 2.377, 2.291, 2.064, 1.971, 1.857, 2.42, 2.568, 2.447, 2.318, 2.072]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "East Anglia One"

class East_Anglia_One_North(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.3757, 5.3484, 5.9716, 7.0348, 5.3957, 5.9152, 8.3566, 16.1889, 15.0673, 9.8276, 7.8423, 6.6758]
        a = [9.16, 8.65, 9.67, 9.73, 9.0, 8.8, 11.28, 14.06, 13.27, 11.7, 10.4, 10.06]
        k = [2.178, 2.236, 2.357, 1.979, 2.025, 1.857, 1.857, 2.432, 2.611, 2.455, 2.553, 2.014]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "East Anglia One North"

class East_Anglia_Two(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.3322, 6.0191, 5.9567, 6.9865, 5.7174, 5.5361, 8.3262, 16.4053, 15.3208, 9.5793, 8.16, 5.6604]
        a = [9.38, 8.75, 9.64, 9.75, 8.92, 8.91, 11.06, 13.8, 13.24, 11.69, 10.59, 9.95]
        k = [2.193, 2.283, 2.385, 2.084, 2.033, 1.893, 1.834, 2.42, 2.584, 2.627, 2.514, 2.201]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "East Anglia Two"

class Egmond_aan_zee(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.7609, 5.8283, 3.7976, 7.2664, 5.8778, 6.5684, 8.093, 13.4552, 16.3199, 10.7228, 8.993, 7.3168]
        a = [7.21, 8.33, 8.62, 9.44, 8.44, 9.21, 10.52, 11.9, 11.16, 10.22, 8.86, 8.31]
        k = [1.854, 2.408, 2.6, 2.271, 2.162, 2.252, 2.299, 2.303, 2.111, 1.943, 1.99, 1.779]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Egmond aan zee"

class Enbw_he_dreiht(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.3813, 4.0997, 5.8114, 7.5774, 7.5083, 5.9935, 7.6353, 12.1128, 14.5519, 12.1249, 9.6342, 8.5693]
        a = [8.87, 7.7, 8.84, 11.55, 11.7, 11.26, 11.65, 13.7, 13.48, 12.68, 11.86, 10.51]
        k = [2.08, 2.064, 1.881, 2.096, 2.377, 2.311, 1.791, 2.393, 2.217, 2.213, 2.369, 2.115]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Enbw he dreiht"

class Eneco_luchterduinen(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.8007, 6.1557, 6.2208, 6.4858, 5.471, 5.4741, 7.7938, 13.2815, 16.8045, 10.4752, 8.6837, 7.3532]
        a = [7.95, 9.0, 9.45, 10.41, 8.87, 9.33, 11.3, 12.62, 12.07, 11.04, 9.49, 9.34]
        k = [2.002, 2.436, 2.662, 2.533, 2.244, 2.291, 2.205, 2.432, 2.26, 2.127, 2.174, 2.068]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Eneco luchterduinen"

class Frederikshavn(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.5093, 5.0937, 6.8156, 7.1568, 6.7205, 8.4943, 9.6987, 7.0914, 11.4876, 19.9499, 8.9685, 4.0137]
        a = [8.19, 8.5, 9.43, 9.66, 10.58, 10.31, 11.05, 11.86, 12.58, 13.59, 12.05, 8.67]
        k = [2.037, 2.111, 2.131, 2.268, 2.393, 2.385, 2.646, 2.936, 2.588, 2.877, 2.307, 2.229]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Frederikshavn"

class Galene(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.0303, 4.2934, 5.5009, 7.5245, 7.3798, 8.4888, 8.8912, 12.0648, 12.8858, 14.6568, 7.8097, 5.4738]
        a = [9.08, 8.46, 9.59, 11.12, 11.82, 11.23, 10.69, 12.56, 12.63, 13.08, 10.49, 8.83]
        k = [1.9, 2.264, 2.346, 2.568, 2.76, 2.115, 1.9, 2.326, 2.342, 2.369, 1.771, 1.744]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Galene"

class Galloper_and_greater_gabbard(UniformWeibullSite):
    """
        Parameters
        ----------
        Site has a mixture of turbine heights (78m and 120m)
        Use an average of 100m to find f,a, and k
    """
    def __init__(self, ti=0.07, shear=None):
        f = [5.4426, 6.118, 6.4553, 6.8242, 5.8321, 5.4334, 8.1888, 16.1154, 15.9656, 9.7803, 7.7309, 6.1135]
        a = [9.54, 9.03, 9.1, 9.74, 8.54, 8.62, 10.82, 13.19, 12.44, 11.22, 9.98, 9.32]
        k = [2.365, 2.65, 2.35, 2.334, 2.084, 1.92, 1.967, 2.639, 2.588, 2.729, 2.42, 2.248]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Galloper and greater gabbard"

class Galloper_and_greater_gabbard2(UniformWeibullSite):
    """
        Parameters
        ----------
        Site has a mixture of turbine heights (78m and 120m)
        Use an average of 100m to find f,a, and k
    """
    def __init__(self, ti=0.07, shear=None):
        f = [5.0209, 6.2021, 6.7925, 6.9918, 5.725, 5.2612, 7.887, 16.5359, 16.1545, 9.7783, 7.2664, 6.3843]
        a = [9.47, 9.22, 9.12, 9.76, 8.68, 8.61, 10.85, 13.1, 12.14, 11.14, 9.91, 9.35]
        k = [2.26, 2.768, 2.396, 2.35, 2.232, 1.873, 2.029, 2.568, 2.502, 2.635, 2.357, 2.205]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Galloper and greater gabbard2"

class Gemini1_and_2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.3623, 4.0746, 5.6296, 7.9884, 7.5715, 5.8825, 7.7741, 12.2362, 14.6049, 12.1738, 9.4111, 8.291]
        a = [8.58, 7.97, 9.23, 10.82, 10.96, 10.43, 11.14, 13.01, 12.44, 12.04, 11.21, 9.66]
        k = [2.15, 2.26, 2.162, 2.17, 2.639, 2.479, 2.025, 2.521, 2.412, 2.33, 2.533, 2.092]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Gemini1 and 2"

class Gemini3(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.4552, 4.0251, 5.3797, 7.9273, 7.3807, 6.0116, 7.819, 12.5416, 15.0398, 11.9543, 9.4486, 8.0172]
        a = [8.8, 7.7, 9.31, 10.82, 11.03, 10.76, 11.92, 14.07, 13.34, 12.84, 11.49, 10.07]
        k = [2.119, 1.959, 2.061, 1.881, 2.342, 2.197, 1.865, 2.42, 2.162, 2.213, 2.268, 2.08]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Gemini3"

class Gode_wind1(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.1863, 4.0582, 6.1288, 8.2388, 7.55, 5.8585, 7.6598, 11.7577, 14.0663, 12.2909, 10.3217, 7.883]
        a = [7.96, 7.8, 9.51, 11.07, 11.3, 10.57, 11.3, 13.08, 12.48, 11.9, 10.72, 10.01]
        k = [2.01, 2.178, 2.205, 2.35, 2.9, 2.666, 2.217, 2.729, 2.373, 2.385, 2.322, 2.146]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Gode wind1"

class Gode_wind2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.1654, 4.0048, 6.0579, 8.2743, 7.5568, 5.77, 7.8014, 11.926, 14.1246, 12.2353, 10.1867, 7.8969]
        a = [7.95, 7.78, 9.5, 11.02, 11.26, 10.52, 11.23, 13.04, 12.5, 11.89, 10.77, 9.91]
        k = [2.014, 2.143, 2.232, 2.334, 2.943, 2.686, 2.213, 2.729, 2.377, 2.369, 2.369, 2.123]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Gode wind2"

class Gode_wind3(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.1988, 4.042, 6.0884, 8.2299, 7.5672, 5.8259, 7.7614, 11.8399, 14.0766, 12.211, 10.3152, 7.8439]
        a = [8.46, 7.45, 9.37, 11.62, 11.79, 10.73, 12.24, 14.28, 13.56, 12.53, 11.26, 10.36]
        k = [1.979, 1.951, 2.018, 2.158, 2.635, 2.166, 1.998, 2.631, 2.236, 2.225, 2.213, 2.037]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Gode wind3"

class Gode_wind4(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [4.238, 4.0102, 6.0324, 8.2892, 7.5518, 5.8221, 7.7362, 12.0239, 13.971, 12.2505, 10.2838, 7.791]
        a = [8.34, 7.44, 9.4, 11.61, 11.78, 10.82, 12.28, 14.28, 13.65, 12.5, 11.24, 10.34]
        k = [1.955, 1.943, 2.006, 2.174, 2.646, 2.193, 2.029, 2.631, 2.268, 2.213, 2.205, 2.037]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Gode wind4"

class Gunfleet_sands(UniformWeibullSite):
    """
        Parameters
        ----------
        Site has a mixture of turbine heights (75m and 84m)
        Used an average of 80m to find f,a, and k
    """
    def __init__(self, ti=0.07, shear=None):
        f = [5.93, 4.6454, 7.0004, 6.5883, 6.0294, 6.2364, 8.5537, 15.3445, 15.6169, 10.921, 6.9747, 6.1593]
        a = [9.55, 9.83, 9.48, 8.96, 8.13, 8.23, 10.78, 12.44, 12.4, 10.5, 10.17, 10.08]
        k = [3.107, 3.049, 2.783, 2.146, 2.127, 1.99, 2.115, 2.549, 2.982, 2.814, 3.029, 2.643]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Gunfleet sands"

class Hollandse_kust_noord(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.2005, 5.7159, 6.5174, 7.088, 5.3943, 5.369, 7.5235, 14.1434, 15.9525, 10.7133, 8.9224, 7.4598]
        a = [8.13, 8.76, 10.2, 10.71, 9.46, 9.91, 11.81, 13.21, 12.79, 11.69, 10.1, 9.46]
        k = [1.947, 2.439, 2.861, 2.373, 2.15, 2.088, 2.084, 2.303, 2.236, 2.088, 2.131, 1.928]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Hollandse kust noord"

class Hollandse_kust_zuid1(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [5.8388, 6.1978, 6.2111, 6.8814, 5.4382, 5.0022, 7.9761, 13.2403, 16.9293, 10.3495, 8.5528, 7.3824]
        a = [8.24, 8.87, 10.0, 10.76, 9.31, 9.91, 11.89, 13.76, 13.11, 11.5, 9.72, 9.78]
        k = [1.924, 2.143, 2.436, 2.271, 1.908, 1.908, 1.896, 2.381, 2.178, 2.049, 1.928, 2.053]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Hollandse kust zuid1"

class Hollandse_kust_zuid2(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [5.9128, 6.2162, 6.2266, 6.1327, 4.738, 6.1562, 8.2606, 12.9555, 17.0292, 10.3537, 8.6145, 7.404]
        a = [8.2, 8.91, 9.95, 11.37, 9.04, 9.83, 12.03, 13.8, 13.06, 11.48, 9.59, 9.67]
        k = [1.908, 2.158, 2.443, 2.443, 1.713, 1.912, 1.955, 2.408, 2.162, 2.045, 1.904, 2.021]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Hollandse kust zuid2"

class Hollandse_kust_zuid3(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [5.8491, 6.2793, 6.2031, 6.828, 5.2294, 5.3236, 7.9968, 13.0656, 17.0699, 10.2728, 8.4967, 7.3856]
        a = [8.32, 8.85, 10.09, 10.75, 9.26, 9.68, 12.07, 13.75, 13.13, 11.55, 9.58, 9.73]
        k = [1.951, 2.146, 2.482, 2.271, 1.885, 1.857, 1.951, 2.396, 2.189, 2.072, 1.9, 2.045]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Hollandse kust zuid3"

class Hollandse_kust_zuid4(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [6.0173, 6.4037, 6.1866, 5.6387, 5.058, 6.4784, 8.2159, 12.6897, 17.3678, 10.2146, 8.4404, 7.2888]
        a = [8.3, 8.87, 10.13, 11.51, 9.05, 10.11, 12.37, 13.75, 13.12, 11.48, 9.43, 9.65]
        k = [1.936, 2.146, 2.623, 2.51, 1.865, 2.014, 2.033, 2.416, 2.186, 2.072, 1.869, 2.029]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Hollandse kust zuid4"

class Hollandse_wind1(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [5.1883, 5.267, 6.7422, 7.063, 5.3511, 5.5233, 7.7934, 14.5474, 15.5923, 10.4235, 9.0174, 7.491]
        a = [8.46, 8.65, 10.08, 10.5, 9.19, 9.64, 11.55, 13.72, 13.36, 12.06, 10.36, 9.94]
        k = [1.908, 2.338, 2.361, 2.143, 1.963, 1.9, 1.826, 2.232, 2.236, 2.088, 2.088, 2.01]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Hollandse kust zuid4"

class Hornsea1_and_2(UniformWeibullSite):
    """
        Parameters
        ----------
        Site has a mixture of turbine heights (120.5m and 110m)
        Used an average of 115m to find f,a, and k
    """
    def __init__(self, ti=0.07, shear=None):
        f = [6.2104, 4.2258, 5.2256, 6.5815, 5.2046, 7.0032, 10.1727, 13.3421, 14.8264, 11.0815, 8.2501, 7.876]
        a = [9.59, 8.73, 9.05, 10.37, 9.31, 10.4, 11.81, 13.09, 12.98, 12.34, 10.36, 10.24]
        k = [2.377, 2.201, 2.545, 2.236, 2.021, 2.35, 1.979, 2.283, 2.643, 2.275, 2.432, 2.15]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Hornsea1 and 2"

class Hornsea3(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [5.7147, 4.3457, 5.3493, 6.963, 5.0718, 6.6634, 9.994, 12.9412, 14.9603, 11.376, 9.009, 7.6116]
        a = [9.83, 8.4, 8.96, 10.46, 10.14, 10.25, 12.33, 14.03, 14.23, 12.96, 11.34, 10.43]
        k = [2.252, 1.904, 2.193, 1.865, 1.986, 1.982, 1.811, 2.076, 2.424, 2.107, 2.393, 2.057]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Hornsea3"

class Hornsea4(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [6.8668, 4.3593, 5.1989, 6.3467, 5.2331, 7.2821, 10.2713, 13.4124, 14.281, 11.3401, 7.4861, 7.9221]
        a = [9.98, 8.9, 9.11, 10.74, 9.72, 11.23, 12.34, 14.44, 14.26, 13.38, 10.92, 10.6]
        k = [2.232, 1.975, 2.197, 1.979, 1.861, 2.107, 1.838, 2.232, 2.561, 2.463, 2.217, 2.1]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Hornsea4"

class Hornsrev(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [3.8916, 3.273, 5.2021, 7.8877, 7.8294, 6.1176, 7.2306, 11.9464, 13.238, 12.6364, 12.5044, 8.2427]
        a = [7.59, 7.64, 8.93, 9.97, 10.6, 9.93, 10.38, 11.43, 11.6, 10.97, 11.05, 9.47]
        k = [2.037, 2.299, 2.314, 2.271, 2.396, 2.283, 2.135, 2.318, 2.248, 2.291, 2.408, 1.939]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Hornsrev"

class Hornsrev2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [3.7867, 3.5145, 5.4229, 7.2521, 7.8752, 6.2086, 7.1417, 12.072, 13.3013, 12.603, 12.2829, 8.539]
        a = [7.38, 7.56, 9.36, 9.88, 10.66, 10.15, 10.76, 11.39, 11.56, 10.97, 11.2, 9.67]
        k = [1.857, 2.209, 2.615, 2.213, 2.381, 2.217, 2.264, 2.314, 2.178, 2.252, 2.42, 1.967]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Hornsrev2"

class Hornsrev3(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [3.7401, 3.5254, 5.4369, 7.4509, 7.5387, 6.2726, 7.2005, 12.0626, 13.2567, 12.6118, 12.521, 8.3828]
        a = [7.98, 8.27, 10.33, 10.84, 11.5, 11.22, 11.59, 12.21, 12.34, 11.82, 12.13, 10.59]
        k = [2.033, 2.424, 2.986, 2.49, 2.682, 2.553, 2.412, 2.443, 2.283, 2.396, 2.557, 2.158]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Hornsrev3"

class Humber_Gateway(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.9475, 5.0958, 5.2537, 6.2226, 5.4231, 7.6087, 9.7889, 12.4779, 14.6657, 13.8074, 8.0427, 4.666]
        a = [9.99, 8.57, 8.59, 9.52, 9.56, 10.35, 11.66, 13.31, 12.74, 11.67, 10.77, 9.77]
        k = [2.494, 2.432, 2.408, 2.24, 2.283, 2.514, 2.229, 2.639, 2.9, 2.537, 3.193, 2.268]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Humber Gateway"

class Hywind_scotland(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.3397, 3.7613, 3.3408, 4.2647, 6.4242, 9.6318, 13.6351, 14.6819, 8.8451, 8.7339, 10.2971, 10.0445] 
        a = [10.11, 7.49, 7.15, 9.49, 10.58, 12.18, 12.68, 14.21, 12.6, 13.75, 13.46, 11.37] 
        k = [1.795, 1.529, 1.521, 1.99, 1.693, 1.963, 1.994, 2.646, 2.408, 2.256, 2.498, 2.154] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Hywind scotland"

class Hywind_tampen(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [9.4311, 9.0292, 4.5082, 3.1007, 4.3193, 12.4677, 16.0805, 10.2249, 8.9191, 8.0158, 6.5309, 7.3725] 
        a = [9.87, 10.26, 7.7, 7.57, 8.83, 14.2, 14.08, 12.77, 11.94, 10.78, 9.56, 9.16] 
        k = [1.916, 2.131, 1.834, 1.725, 1.74, 1.854, 2.221, 2.295, 1.947, 2.045, 1.666, 1.615] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Hywind tampen"

class Ijmuiden_ver_alpha(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [5.2324, 4.7822, 6.4415, 7.1478, 5.3775, 5.7478, 8.1596, 14.8061, 14.9453, 10.5949, 9.1156, 7.6491]
        a = [8.76, 8.49, 9.87, 10.45, 9.2, 9.4, 11.48, 14.0, 13.5, 12.2, 10.48, 10.06]
        k = [2.002, 2.209, 2.268, 2.068, 1.99, 1.916, 1.775, 2.221, 2.291, 2.061, 2.15, 1.998]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Ijmuiden ver alpha"

class Ijmuiden_ver_alpha2(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [5.1698, 4.8681, 6.5821, 7.2163, 5.3499, 5.6375, 8.0149, 14.6794, 15.17, 10.5879, 9.0861, 7.638]
        a = [8.64, 8.5, 10.02, 10.42, 9.11, 9.69, 11.47, 13.88, 13.44, 12.19, 10.46, 10.02]
        k = [1.955, 2.252, 2.338, 2.084, 1.955, 1.998, 1.783, 2.213, 2.283, 2.057, 2.146, 2.006]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Ijmuiden ver alpha2"

class Ijmuiden_ver_beta(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [5.0466, 4.7634, 6.4065, 7.2789, 5.4388, 5.5806, 8.3672, 14.5867, 14.9377, 10.7242, 9.2403, 7.629]
        a = [8.71, 8.52, 9.84, 10.37, 9.09, 9.66, 11.47, 14.03, 13.44, 12.37, 10.57, 10.09]
        k = [1.982, 2.232, 2.252, 2.049, 1.893, 1.955, 1.795, 2.221, 2.252, 2.061, 2.186, 2.01]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Ijmuiden ver beta"

class Ijmuiden_ver_beta2(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [5.0404, 4.8305, 6.6078, 7.2861, 5.3979, 5.4986, 8.2098, 14.4744, 15.2467, 10.6564, 9.0991, 7.6525]
        a = [8.59, 8.51, 9.98, 10.39, 9.14, 9.71, 11.46, 13.89, 13.4, 12.25, 10.53, 10.03]
        k = [1.932, 2.271, 2.307, 2.08, 1.912, 1.951, 1.795, 2.209, 2.252, 2.057, 2.166, 1.998]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Ijmuiden ver beta2"

class Ijmuiden_ver_noord_and_lagelander_zuid(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [4.8968, 4.7516, 6.5439, 7.432, 5.4705, 5.4232, 8.2944, 14.2649, 15.3154, 10.8197, 9.1161, 7.6716] 
        a = [8.74, 8.35, 9.98, 10.41, 9.08, 9.81, 11.6, 13.87, 13.37, 12.3, 10.67, 9.97] 
        k = [1.959, 2.225, 2.295, 2.033, 1.854, 1.967, 1.842, 2.189, 2.229, 2.021, 2.186, 1.986]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Ijmuiden ver noord & lagelander zuid"
        

class Inner_dowsing_lincs(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.1275, 5.26, 5.6422, 6.5896, 6.0654, 5.8534, 9.3545, 16.405, 13.3019, 11.3412, 8.1635, 5.8958]
        a = [9.74, 8.45, 8.32, 9.79, 9.14, 10.28, 11.34, 12.79, 13.06, 11.74, 10.75, 10.5]
        k = [2.404, 2.338, 2.291, 2.467, 2.275, 2.764, 2.053, 2.498, 3.127, 2.752, 2.963, 2.74]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Inner Dowsing Lincs"

class Jammerland_bugt(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [3.9609, 3.2889, 4.9113, 6.2683, 8.375, 8.0219, 10.0765, 11.122, 13.8942, 14.7601, 10.8753, 4.4455]
        a = [8.41, 8.24, 10.49, 12.4, 12.61, 11.05, 10.61, 13.34, 13.91, 13.1, 11.9, 9.51]
        k = [1.932, 2.271, 2.373, 2.854, 2.307, 2.283, 1.928, 2.725, 2.654, 2.58, 2.236, 1.967]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Jammerland bugt" 

class Kaskasi(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [4.309, 3.6525, 5.8296, 8.2841, 7.481, 6.1692, 6.9043, 11.5917, 14.0924, 12.5205, 11.2644, 7.9013]
        a = [8.52, 7.38, 9.7, 12.14, 11.39, 11.14, 11.63, 13.97, 13.4, 12.39, 11.73, 10.61]
        k = [1.912, 2.057, 2.139, 2.525, 2.42, 2.264, 1.904, 2.611, 2.162, 2.189, 2.303, 2.045]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "kaskasi"

class Kattegat_offshore(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [3.7746, 2.9613, 4.6827, 7.3809, 10.2756, 8.8353, 8.2292, 11.6975, 12.8569, 14.4438, 8.7038, 6.1584]
        a = [9.41, 8.83, 9.76, 11.93, 12.44, 11.32, 10.47, 12.43, 12.66, 12.77, 9.94, 8.6]
        k = [2.154, 2.412, 2.721, 2.721, 3.229, 2.213, 1.932, 2.244, 2.377, 2.236, 1.701, 1.697]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Kattegat offshore"

class Kattegat_syd(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [4.8169, 3.6146, 4.7151, 7.385, 7.8111, 7.8299, 8.3149, 12.8599, 13.1645, 14.9112, 9.0843, 5.4926]
        a = [8.95, 8.15, 9.35, 11.22, 11.83, 11.17, 10.77, 12.67, 12.89, 13.13, 10.42, 8.6]
        k = [1.916, 2.342, 2.338, 2.619, 2.604, 2.17, 2.002, 2.338, 2.432, 2.369, 1.732, 1.666]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Kattegat syd"


class Kattegat_syd2(UniformWeibullSite):
    """
        Parameters
        ----------
        height not given in sea impact or wind turbine data
    """
    def __init__(self, ti=0.07, shear=None):
        f = [4.8169, 3.6146, 4.7151, 7.385, 7.8111, 7.8299, 8.3149, 12.8599, 13.1645, 14.9112, 9.0843, 5.4926]
        a = [8.95, 8.15, 9.35, 11.22, 11.83, 11.17, 10.77, 12.67, 12.89, 13.13, 10.42, 8.6]
        k = [1.916, 2.342, 2.338, 2.619, 2.604, 2.17, 2.002, 2.338, 2.432, 2.369, 1.732, 1.666]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Kattegat syd2"

class Kentish_flats(UniformWeibullSite):
    """
        Parameters
        ----------
        Site has a mixture of turbine heights (70m and 84m)
        Used an average of 77m to find f,a, and k
    """
    def __init__(self, ti=0.07, shear=None):
        f = [4.4752, 4.5156, 7.2211, 7.6845, 6.0358, 4.4077, 6.746, 15.426, 18.0823, 11.3142, 8.2647, 5.8269]
        a = [8.95, 9.47, 9.02, 8.37, 8.14, 9.06, 10.96, 12.65, 12.21, 9.76, 9.31, 9.3]
        k = [2.404, 2.818, 2.471, 2.08, 2.553, 2.381, 2.217, 2.42, 2.646, 2.693, 2.451, 2.377]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Kentish flats"

class Lagelander_noord(UniformWeibullSite):
    """
        Parameters
        ----------
        Site has a mixture of turbine heights (75m and 84m)
        Used an average of 80m to find f,a, and k
    """
    def __init__(self, ti=0.07, shear=None):
        f = [3.9913, 2.3237, 4.7102, 6.4711, 11.1528, 7.02, 4.812, 12.3752, 12.8529, 14.7034, 12.6003, 6.9872]
        a = [8.55, 8.3, 10.05, 11.77, 11.69, 10.55, 10.31, 14.31, 13.12, 13.12, 13.82, 11.51]
        k = [1.947, 2.303, 2.096, 2.6, 2.49, 2.268, 1.826, 3.354, 2.287, 2.627, 2.729, 2.342]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Lagelander noord"

class Lillebaelt_vind(UniformWeibullSite):
    """
        Parameters
        ----------
        Site has a mixture of turbine heights (75m and 84m)
        Used an average of 80m to find f,a, and k
    """
    def __init__(self, ti=0.07, shear=None):
        f = [3.9913, 2.3237, 4.7102, 6.4711, 11.1528, 7.02, 4.812, 12.3752, 12.8529, 14.7034, 12.6003, 6.9872]
        a = [8.55, 8.3, 10.05, 11.77, 11.69, 10.55, 10.31, 14.31, 13.12, 13.12, 13.82, 11.51]
        k = [1.947, 2.303, 2.096, 2.6, 2.49, 2.268, 1.826, 3.354, 2.287, 2.627, 2.729, 2.342]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Lillebaelt vind"

class Lincs_Lynn(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.2458, 5.1273, 5.6712, 6.6116, 6.1401, 5.9967, 8.7427, 16.9688, 14.0865, 11.1005, 7.804, 5.5048]
        a = [9.8, 8.51, 8.34, 9.82, 9.2, 10.52, 11.51, 12.79, 12.8, 11.55, 10.73, 10.41]
        k = [2.447, 2.354, 2.307, 2.455, 2.287, 3.088, 2.084, 2.428, 3.143, 2.604, 3.045, 2.771]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Lincs Lynn"
        
class London_array(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.6872, 5.5338, 6.751, 6.948, 5.8429, 5.7129, 9.3519, 13.9453, 17.5004, 9.8699, 7.4342, 6.4224]
        a = [9.12, 9.8, 9.24, 9.36, 8.47, 8.29, 10.78, 12.5, 12.35, 10.72, 9.85, 9.31]
        k = [2.33, 3.033, 2.545, 2.232, 2.236, 1.885, 2.111, 2.467, 3.002, 2.674, 2.529, 2.26]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "London array"

class Lovstaviken(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.2927, 3.0888, 5.4044, 8.0007, 7.3752, 11.9946, 8.9107, 10.7572, 12.8737, 13.9424, 9.204, 4.1554]
        a = [9.03, 8.05, 9.3, 9.95, 10.7, 9.45, 8.39, 10.22, 10.25, 10.55, 8.42, 7.92]
        k = [2.576, 2.67, 2.674, 2.982, 2.971, 2.482, 1.85, 2.244, 2.4, 2.166, 1.748, 2.209]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Lovstaviken"     

class Mareld(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.0825, 5.7796, 8.3725, 8.1699, 5.6258, 7.1736, 9.4407, 11.347, 17.2519, 13.14, 5.1396, 3.4769] 
        a = [8.84, 8.68, 10.29, 10.96, 9.72, 10.81, 11.5, 11.79, 12.58, 14.02, 11.49, 8.75] 
        k = [1.834, 1.975, 2.279, 2.436, 2.104, 1.979, 1.986, 2.076, 2.068, 2.307, 2.229, 1.764]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Mareld"  

class Meerwind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.2737, 3.7014, 6.1133, 8.1687, 7.5336, 6.2348, 6.9346, 11.5551, 14.089, 12.3608, 11.2644, 7.7705] 
        a = [8.38, 7.47, 9.68, 11.26, 11.14, 10.7, 11.1, 13.04, 12.58, 11.74, 11.03, 9.97] 
        k = [2.311, 2.24, 2.314, 2.604, 2.662, 2.6, 2.205, 2.717, 2.443, 2.314, 2.471, 2.092]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Meerwind"  

class Merkur(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.1103, 4.0369, 5.8913, 8.2812, 7.4713, 6.046, 7.5001, 12.0147, 14.3301, 12.32, 9.7408, 8.2573] 
        a = [8.11, 7.91, 9.51, 10.94, 11.24, 10.56, 11.13, 13.32, 12.39, 11.98, 11.0, 9.83] 
        k = [2.014, 2.201, 2.256, 2.26, 2.924, 2.654, 2.096, 2.854, 2.393, 2.357, 2.451, 2.127]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Merkur"  

class N_91(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.6693, 4.0856, 5.7105, 7.5153, 7.1346, 6.1584, 7.8238, 12.3861, 14.6073, 12.0113, 9.4006, 8.4972] 
        a = [9.12, 7.59, 8.93, 11.32, 11.86, 11.28, 11.71, 13.56, 13.51, 12.62, 11.99, 10.5] 
        k = [2.178, 1.975, 1.959, 2.053, 2.357, 2.291, 1.803, 2.318, 2.201, 2.17, 2.4, 2.064] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "N-9.1"

class N_92(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.712, 4.1229, 5.6957, 7.4804, 7.0093, 6.2478, 7.8933, 12.3821, 14.6786, 11.8584, 9.4231, 8.4963] 
        a = [9.12, 7.56, 8.95, 11.26, 11.89, 11.32, 11.75, 13.55, 13.52, 12.63, 11.99, 10.52] 
        k = [2.166, 1.9, 1.959, 2.041, 2.365, 2.295, 1.814, 2.318, 2.205, 2.162, 2.385, 2.072] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "N-9.2"

class N_93(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.6761, 4.1365, 5.5434, 7.5583, 7.0053, 6.2483, 7.7464, 12.3358, 14.5449, 12.0482, 9.5021, 8.6546] 
        a = [8.99, 7.41, 9.03, 11.42, 11.94, 11.49, 11.6, 13.57, 13.55, 12.55, 12.05, 10.63] 
        k = [2.127, 1.916, 1.967, 2.072, 2.404, 2.326, 1.783, 2.318, 2.221, 2.166, 2.389, 2.1]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "N-9.3"

class Nederwiek_zuid(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.3427, 4.4568, 6.1118, 7.0196, 5.4485, 5.8993, 8.882, 14.5736, 14.471, 10.9385, 9.195, 7.6612] 
        a = [9.08, 8.51, 9.56, 10.6, 9.17, 9.44, 11.41, 14.36, 13.66, 12.36, 10.68, 10.32] 
        k = [2.107, 2.186, 2.201, 2.092, 1.854, 1.92, 1.736, 2.182, 2.373, 2.049, 2.221, 2.084] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nederwiek zuid"

class Nissum_bredning_vind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [2.3905, 2.0534, 6.5897, 8.7204, 8.0794, 3.9488, 8.2755, 12.8441, 12.9241, 13.5158, 15.1512, 5.5071] 
        a = [7.99, 8.57, 10.68, 10.8, 10.75, 10.5, 12.33, 11.93, 12.02, 11.74, 12.51, 10.37] 
        k = [2.658, 2.744, 2.982, 2.393, 2.592, 3.002, 3.275, 2.396, 2.26, 2.252, 2.436, 1.893] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nissum bredning vind"

class Nobelwind_and_belwind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.6273, 6.6891, 7.044, 7.1303, 5.471, 4.6198, 7.9444, 12.7723, 19.3098, 9.6196, 7.4495, 6.3229] 
        a = [8.32, 8.17, 8.72, 8.95, 7.93, 7.93, 10.79, 11.74, 11.49, 10.2, 8.73, 8.4] 
        k = [2.139, 2.291, 2.361, 2.076, 2.248, 1.783, 2.139, 2.342, 2.201, 2.307, 1.865, 1.838]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nobelwind and belwind"

class Noordhinder_noord(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.4624, 6.7106, 7.1371, 7.0843, 5.4877, 4.8635, 7.7399, 13.3305, 19.0965, 9.5153, 7.3826, 6.1895] 
        a = [9.41, 8.75, 9.69, 10.14, 9.1, 8.92, 12.46, 13.86, 13.49, 11.21, 10.0, 9.54] 
        k = [2.115, 2.326, 2.268, 2.135, 2.115, 1.799, 2.037, 2.506, 2.314, 2.221, 2.045, 1.967] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Noordhinder noord"

class Noordhinder_zuid_and_fairy_bank(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.3352, 6.5503, 7.4771, 7.0944, 5.3873, 4.7951, 7.7763, 12.8495, 19.909, 9.4125, 7.333, 6.0804] 
        a = [9.49, 8.87, 9.5, 10.24, 9.21, 8.96, 12.7, 13.96, 13.6, 11.07, 9.95, 9.25] 
        k = [2.115, 2.342, 2.205, 2.209, 2.193, 1.783, 2.092, 2.592, 2.357, 2.217, 2.053, 1.893]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Noordhinder zuid and fairy bank"

class Nordborg(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [3.2552, 2.2852, 4.9463, 8.4189, 10.1223, 5.2067, 5.5219, 10.0642, 15.9676, 14.055, 13.4526, 6.704] 
        a = [8.74, 8.11, 10.07, 12.03, 11.27, 10.63, 10.96, 14.16, 14.84, 12.58, 12.95, 10.74] 
        k = [2.166, 2.146, 2.053, 2.764, 2.326, 2.303, 2.033, 2.857, 2.787, 2.373, 2.678, 2.291] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nordborg"

class Nordlicht_1(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.3211, 4.0554, 5.874, 7.6816, 7.5575, 5.9993, 7.6177, 12.0641, 14.5047, 12.171, 9.7108, 8.4427] 
        a = [8.88, 7.75, 8.85, 11.5, 11.66, 11.26, 11.74, 13.78, 13.42, 12.7, 11.8, 10.43] 
        k = [2.092, 2.057, 1.865, 2.064, 2.404, 2.318, 1.811, 2.416, 2.209, 2.232, 2.357, 2.096] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nordlicht 1"

class Nordlicht_1_2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.3753, 4.0293, 5.8718, 7.6237, 7.501, 5.9901, 7.6388, 12.1334, 14.5609, 12.1431, 9.5794, 8.5531] 
        a = [8.91, 7.72, 8.85, 11.49, 11.67, 11.3, 11.68, 13.67, 13.47, 12.69, 11.84, 10.44] 
        k = [2.107, 2.061, 1.881, 2.072, 2.389, 2.346, 1.795, 2.377, 2.213, 2.209, 2.373, 2.104]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nordlicht 1 2"

class Nordlicht_1_3(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.3559, 4.0484, 5.8604, 7.6262, 7.5081, 6.0179, 7.6178, 12.1237, 14.5486, 12.1478, 9.6023, 8.543] 
        a = [8.91, 7.73, 8.85, 11.51, 11.68, 11.28, 11.7, 13.68, 13.47, 12.69, 11.83, 10.44] 
        k = [2.107, 2.064, 1.877, 2.076, 2.393, 2.33, 1.799, 2.385, 2.213, 2.213, 2.373, 2.104] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nordlicht 1 3"

class Nordlicht_1_4(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.3374, 4.0876, 5.838, 7.6757, 7.5335, 5.9727, 7.6065, 12.0981, 14.5164, 12.1223, 9.7162, 8.4955] 
        a = [8.86, 7.76, 8.8, 11.55, 11.7, 11.26, 11.71, 13.75, 13.44, 12.7, 11.8, 10.49] 
        k = [2.068, 2.084, 1.854, 2.096, 2.393, 2.314, 1.807, 2.424, 2.213, 2.229, 2.346, 2.107] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nordlicht 1 4"

class Nordlicht_1_5(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.3277, 4.1375, 5.7972, 7.7101, 7.5467, 5.9937, 7.5832, 12.0248, 14.5061, 12.1294, 9.7895, 8.4541] 
        a = [8.85, 7.77, 8.79, 11.58, 11.71, 11.23, 11.76, 13.78, 13.44, 12.69, 11.79, 10.48] 
        k = [2.064, 2.1, 1.846, 2.115, 2.4, 2.311, 1.818, 2.428, 2.217, 2.225, 2.342, 2.096] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nordlicht 1 5"

class Nordlicht_1_6(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.2849, 4.1033, 5.8153, 7.8023, 7.5874, 5.9928, 7.5563, 11.9828, 14.5191, 12.1107, 9.8079, 8.4371] 
        a = [8.84, 7.77, 8.81, 11.54, 11.71, 11.17, 11.83, 13.83, 13.41, 12.69, 11.81, 10.43] 
        k = [2.068, 2.072, 1.85, 2.092, 2.436, 2.295, 1.834, 2.432, 2.221, 2.244, 2.365, 2.088]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nordlicht 1 6"

class Nordlicht_2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.4468, 3.9997, 5.798, 7.7026, 7.4639, 5.8797, 7.7212, 12.2748, 14.5868, 12.1132, 9.4544, 8.5589] 
        a = [8.98, 7.74, 8.91, 11.4, 11.63, 11.39, 11.56, 13.63, 13.5, 12.69, 11.87, 10.36] 
        k = [2.123, 2.061, 1.908, 2.029, 2.377, 2.373, 1.771, 2.35, 2.213, 2.205, 2.377, 2.072] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nordlicht 2"

class Nordsee_ost(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.3099, 3.6516, 5.9496, 8.2282, 7.5522, 6.1645, 6.9335, 11.5348, 14.1305, 12.4218, 11.3194, 7.8041] 
        a = [8.37, 7.47, 9.68, 11.33, 11.06, 10.77, 11.1, 13.02, 12.56, 11.73, 11.07, 10.01] 
        k = [2.326, 2.209, 2.342, 2.627, 2.619, 2.611, 2.193, 2.701, 2.42, 2.314, 2.475, 2.092] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nordsee ost"

class Nordsee1(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.0588, 4.0194, 5.8842, 8.3453, 7.5037, 5.825, 7.7757, 12.1218, 14.2068, 12.402, 9.9112, 7.9462] 
        a = [8.07, 7.77, 9.46, 10.91, 11.33, 10.43, 11.12, 13.03, 12.52, 11.86, 10.8, 9.79] 
        k = [2.041, 2.154, 2.221, 2.295, 3.193, 2.686, 2.162, 2.744, 2.373, 2.338, 2.365, 2.104] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nordsee1"

class Nordsee2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.1038, 4.0148, 6.0426, 8.2725, 7.5094, 5.9051, 7.5631, 11.9731, 14.2234, 12.3057, 10.015, 8.0715] 
        a = [8.44, 7.53, 9.37, 11.37, 11.73, 10.85, 12.25, 14.22, 13.45, 12.63, 11.41, 10.3] 
        k = [1.947, 1.959, 1.986, 2.064, 2.549, 2.221, 1.99, 2.576, 2.217, 2.232, 2.252, 2.057] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nordsee2"

class Nordsee3(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.1219, 4.0257, 5.9962, 8.3037, 7.5486, 5.8325, 7.7001, 11.9341, 14.2148, 12.2802, 10.0852, 7.9569] 
        a = [8.43, 7.45, 9.41, 11.43, 11.73, 10.84, 12.23, 14.22, 13.5, 12.64, 11.32, 10.29] 
        k = [1.963, 1.932, 2.025, 2.084, 2.576, 2.205, 1.994, 2.588, 2.229, 2.24, 2.225, 2.045] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nordsee3"

class Nordsee4(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.1252, 4.0172, 6.0732, 8.2996, 7.5344, 5.822, 7.686, 11.8599, 14.1677, 12.2939, 10.1594, 7.9615] 
        a = [8.46, 7.48, 9.34, 11.49, 11.76, 10.79, 12.24, 14.21, 13.48, 12.62, 11.32, 10.32] 
        k = [1.959, 1.939, 1.99, 2.1, 2.588, 2.182, 1.994, 2.58, 2.221, 2.24, 2.225, 2.049] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nordsee4"

class Nordsee5(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.1398, 4.027, 6.1456, 8.2283, 7.5467, 5.8709, 7.6334, 11.7941, 14.1373, 12.3006, 10.2263, 7.95] 
        a = [8.49, 7.53, 9.28, 11.55, 11.8, 10.76, 12.24, 14.19, 13.49, 12.59, 11.34, 10.36] 
        k = [1.967, 1.971, 1.951, 2.119, 2.604, 2.178, 1.994, 2.568, 2.225, 2.24, 2.229, 2.045] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nordsee5"

class Norfolk_boreas(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.5307, 4.3781, 6.101, 6.8876, 5.3224, 6.1562, 9.0358, 14.8655, 14.1242, 10.8676, 9.0199, 7.7109] 
        a = [9.11, 8.52, 9.5, 10.7, 9.26, 9.45, 11.42, 14.44, 13.76, 12.35, 10.62, 10.3] 
        k = [2.143, 2.15, 2.201, 2.135, 1.896, 1.928, 1.744, 2.174, 2.471, 2.088, 2.209, 2.049] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Norfolk boreas"

class Norfolk_vanguard_and_east_anglia_three(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.5198, 4.8379, 6.2696, 6.773, 5.4724, 5.8469, 8.6234, 15.3101, 14.5194, 10.3121, 8.7564, 7.7589] 
        a = [8.99, 8.47, 9.71, 10.6, 9.21, 9.0, 11.26, 14.33, 13.7, 12.25, 10.72, 10.08] 
        k = [2.072, 2.178, 2.236, 2.111, 1.979, 1.822, 1.717, 2.221, 2.432, 2.186, 2.295, 1.959] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Norfolk vanguard and east anglia three"

class Norfolk_vanguard(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.9535, 4.4466, 6.0586, 6.6504, 5.3687, 6.4215, 9.742, 14.7881, 13.869, 10.321, 8.6738, 7.707] 
        a = [9.13, 8.61, 9.49, 10.67, 9.32, 9.44, 11.28, 14.8, 13.75, 12.48, 10.78, 10.26] 
        k = [2.111, 2.201, 2.217, 2.143, 1.939, 1.963, 1.729, 2.334, 2.611, 2.232, 2.299, 1.998] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Norfolk vanguard"

class Norther(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.553, 7.4046, 7.0172, 7.1611, 5.2356, 4.2929, 8.2863, 12.4759, 18.5651, 10.3056, 7.4293, 6.2735] 
        a = [8.83, 8.9, 9.54, 9.97, 8.69, 9.04, 12.05, 12.24, 12.6, 10.15, 9.17, 9.05] 
        k = [2.35, 2.439, 2.678, 2.799, 2.525, 2.123, 2.521, 2.65, 2.455, 2.096, 1.947, 1.986] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Norther"

class Northwester2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.691, 6.6454, 6.9721, 7.0945, 5.5731, 4.681, 7.8707, 13.1421, 19.0414, 9.5455, 7.4749, 6.2683] 
        a = [9.09, 8.87, 9.47, 9.81, 8.63, 8.62, 11.61, 12.69, 12.33, 10.83, 9.55, 9.29] 
        k = [2.357, 2.506, 2.549, 2.264, 2.479, 1.92, 2.229, 2.502, 2.322, 2.275, 2.045, 2.037] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Northerwester2"

class Northwind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.5702, 6.9894, 6.9377, 7.1691, 5.4535, 4.4457, 8.0971, 12.2892, 19.3547, 9.9101, 7.4097, 6.3735] 
        a = [8.26, 8.16, 8.76, 9.19, 7.97, 7.99, 10.88, 11.54, 11.66, 9.63, 8.67, 8.32] 
        k = [2.15, 2.244, 2.416, 2.373, 2.295, 1.803, 2.217, 2.342, 2.264, 1.979, 1.85, 1.826] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Northwind"

class Nysted(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [3.1998, 2.9723, 5.7985, 9.19, 8.6391, 5.9633, 6.3064, 11.4093, 13.8516, 15.9107, 11.1522, 5.6069]
        a = [7.55, 6.85, 9.26, 9.78, 9.19, 8.82, 9.3, 10.7, 11.89, 11.24, 10.36, 7.84]
        k = [2.053, 2.178, 2.037, 2.381, 2.307, 2.334, 2.104, 2.494, 2.646, 2.381, 2.518, 1.975]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Nysted"

class Oranje_wind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.3859, 5.3957, 6.5498, 7.0654, 5.4017, 5.6431, 7.6577, 14.6321, 15.7201, 10.2397, 8.7835, 7.5253] 
        a = [8.54, 8.58, 10.12, 10.41, 9.26, 9.33, 11.71, 13.72, 13.42, 11.88, 10.27, 9.99] 
        k = [1.971, 2.217, 2.4, 2.104, 2.064, 1.838, 1.857, 2.236, 2.26, 2.1, 2.076, 2.033] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Oranje Wind"

class Outer_Dowsing(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.3565, 4.1167, 5.2406, 6.4653, 5.2082, 8.0198, 9.5278, 13.8556, 14.7805, 11.5022, 7.7816, 7.1454]
        a = [10.07, 8.75, 9.31, 10.42, 9.33, 11.04, 12.63, 14.31, 14.57, 12.84, 10.98, 10.78]
        k = [2.326, 2.088, 2.439, 1.994, 1.818, 2.209, 1.951, 2.248, 2.666, 2.283, 2.424, 2.174]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Outer Dowsing"

class Paludan_flak(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [3.0325, 3.2412, 5.7251, 7.2313, 7.368, 9.0787, 7.4357, 11.7864, 15.6297, 15.2099, 10.095, 4.1665] 
        a = [8.74, 8.1, 9.41, 11.08, 12.09, 10.9, 10.87, 14.03, 13.8, 12.85, 12.61, 10.36] 
        k = [2.068, 2.189, 2.123, 2.588, 2.432, 2.252, 1.979, 3.162, 2.607, 2.525, 2.424, 2.154] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Paludan Flak"

class Prinses_amaliawindpark(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.4484, 5.9856, 6.2154, 7.2873, 5.2432, 5.3634, 7.4566, 13.773, 16.4265, 10.579, 8.8316, 7.39] 
        a = [7.91, 9.02, 9.62, 10.38, 8.82, 9.37, 11.35, 12.65, 12.13, 11.13, 9.65, 9.29] 
        k = [2.01, 2.623, 2.725, 2.463, 2.229, 2.217, 2.174, 2.373, 2.264, 2.1, 2.115, 2.014] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Prinses Amaliawindpark"

class Race_Bank(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.6315, 4.7266, 5.5268, 6.3947, 6.4593, 6.8263, 8.5067, 14.6627, 16.0396, 10.8766, 8.0631, 5.2862]
        a = [9.78, 8.57, 8.61, 9.8, 9.3, 10.39, 11.58, 13.15, 12.89, 11.38, 10.45, 9.41]
        k = [2.346, 2.381, 2.393, 2.287, 2.264, 2.623, 2.119, 2.436, 2.943, 2.455, 2.803, 2.025]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Race Bank"

class Rentel(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.5598, 7.1644, 6.9735, 7.1305, 5.4546, 4.3391, 8.1902, 12.2368, 19.1618, 10.0095, 7.4357, 6.344] 
        a = [8.95, 8.86, 9.53, 10.16, 8.66, 8.86, 11.83, 12.29, 12.6, 10.35, 9.34, 9.14] 
        k = [2.361, 2.436, 2.654, 2.873, 2.537, 2.018, 2.408, 2.502, 2.42, 2.131, 2.002, 2.01] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Rentel"

class Revolutionwind_southforkwind(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [7.2913, 7.2204, 6.3564, 5.5052, 4.743, 4.7018, 7.7244, 11.6506, 13.331, 11.079, 10.9413, 9.4554] 
        a = [10.37, 10.58, 9.66, 9.33, 9.68, 10.57, 11.77, 13.87, 12.79, 12.12, 12.36, 10.3] 
        k = [2.053, 1.729, 1.635, 1.689, 1.412, 1.42, 1.529, 1.943, 2.076, 2.197, 2.295, 2.201] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Revolutionwind Southforkwind"

class Riffgat(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.055, 4.1089, 6.0668, 7.6451, 7.6815, 6.0533, 7.4568, 13.6643, 13.605, 12.4702, 9.4647, 7.7286] 
        a = [7.79, 7.89, 9.37, 10.84, 10.63, 9.9, 11.4, 12.8, 12.72, 11.75, 10.65, 9.6] 
        k = [2.006, 2.369, 2.232, 2.365, 3.002, 2.639, 2.334, 2.775, 2.4, 2.311, 2.369, 2.158] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Riffgat"

class Rodsand_2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [3.8346, 2.7958, 5.1595, 9.3653, 8.6756, 6.0387, 6.8486, 11.2354, 13.4948, 15.3046, 12.1644, 5.0826]
        a = [7.47, 7.57, 9.56, 9.83, 9.23, 8.57, 9.21, 10.78, 11.9, 11.3, 9.96, 8.15]
        k = [2.26, 2.307, 2.248, 2.357, 2.311, 2.221, 2.045, 2.615, 2.697, 2.357, 2.205, 2.1]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Rodsand 2"
        
class Ronland(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [2.5415, 1.9916, 6.7523, 8.5622, 7.616, 4.2638, 8.3955, 13.0052, 12.8161, 13.4121, 15.1079, 5.5357] 
        a = [7.74, 8.7, 10.63, 10.84, 10.81, 10.6, 12.5, 11.99, 12.17, 11.88, 12.57, 10.46] 
        k = [2.463, 2.998, 2.951, 2.436, 2.604, 2.959, 3.318, 2.424, 2.342, 2.307, 2.443, 1.912] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Ronland"        

class Samso(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [3.0784, 3.1101, 5.7908, 7.2422, 7.4043, 9.0333, 7.4837, 11.5794, 15.8771, 15.0268, 10.1878, 4.1861] 
        a = [7.77, 7.23, 8.53, 9.59, 9.87, 9.5, 9.56, 11.31, 11.47, 10.76, 10.42, 8.48] 
        k = [2.291, 2.084, 2.236, 3.064, 2.049, 2.377, 2.412, 2.807, 2.76, 2.455, 2.291, 2.326] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Samso"

class Sandbank(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.1483, 3.5982, 5.1867, 7.5074, 7.4554, 6.0849, 7.4759, 12.225, 13.7667, 12.3496, 10.9553, 9.2466]
        a = [7.47, 7.29, 8.84, 10.19, 10.69, 10.35, 10.68, 11.72, 11.76, 10.93, 10.99, 9.56]
        k = [1.744, 1.979, 2.123, 2.201, 2.365, 2.256, 2.057, 2.357, 2.217, 2.197, 2.486, 1.928]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Sandbank"

class Scroby_sands(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [7.0549, 5.1622, 5.5022, 6.5119, 5.2475, 7.0114, 10.8945, 7.6868, 14.5083, 13.0827, 10.1843, 7.1534] 
        a = [8.35, 7.89, 8.33, 8.75, 7.88, 8.22, 11.07, 12.86, 12.13, 11.08, 10.64, 8.99] 
        k = [2.291, 2.271, 2.201, 2.072, 1.814, 2.014, 2.225, 2.838, 2.756, 2.486, 2.822, 1.975] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Scroby Sands"

class Seamade(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.766, 6.5803, 6.936, 7.0527, 5.6241, 4.7323, 7.8424, 13.4047, 18.758, 9.5505, 7.4936, 6.2594] 
        a = [9.06, 8.86, 9.49, 9.86, 8.62, 8.64, 11.54, 12.75, 12.31, 10.89, 9.53, 9.35] 
        k = [2.338, 2.502, 2.549, 2.295, 2.494, 1.916, 2.197, 2.518, 2.314, 2.295, 2.033, 2.045] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Seamade"

class Seamade_2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.6108, 6.854, 6.9494, 7.1705, 5.4692, 4.5128, 8.0254, 12.4965, 19.3745, 9.7676, 7.4064, 6.3629] 
        a = [9.05, 8.9, 9.53, 9.9, 8.67, 8.7, 11.69, 12.52, 12.43, 10.62, 9.74, 9.16] 
        k = [2.365, 2.482, 2.635, 2.451, 2.49, 1.959, 2.334, 2.482, 2.357, 2.193, 2.197, 2.018] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Seamade 2"

class Sheringham_shoal(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.5738, 4.4985, 5.399, 6.646, 6.2839, 7.4912, 8.5718, 12.7193, 16.5149, 11.4984, 7.7657, 6.0376] 
        a = [9.68, 8.57, 8.91, 9.68, 8.97, 10.55, 11.88, 13.35, 12.62, 11.61, 10.23, 9.34] 
        k = [2.459, 2.385, 2.553, 2.209, 2.17, 2.764, 2.24, 2.639, 2.697, 2.412, 2.635, 2.002] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Sheringham Shoal"

class Sofia(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.902, 4.2095, 4.8608, 6.5577, 5.9328, 6.7467, 10.1328, 13.0999, 13.6491, 11.0883, 9.5297, 8.2906] 
        a = [9.65, 8.11, 9.22, 11.0, 10.85, 11.36, 12.35, 13.93, 14.01, 13.07, 11.52, 10.68] 
        k = [2.076, 1.717, 2.174, 2.037, 1.998, 2.264, 1.752, 1.986, 2.268, 2.139, 2.154, 2.014] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Sofia"

class Sprogo(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.2622, 3.0457, 4.8975, 6.4104, 9.3711, 8.0175, 8.5156, 11.8478, 13.1271, 14.0204, 11.3484, 5.1363] 
        a = [7.42, 7.43, 9.61, 10.44, 11.36, 9.39, 9.7, 11.95, 12.48, 12.29, 11.34, 9.04] 
        k = [2.029, 2.131, 3.064, 2.721, 2.842, 2.213, 2.275, 2.975, 3.033, 2.807, 2.76, 2.252] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Sprogo"

class Thanet(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.5225, 6.1793, 6.8833, 7.2031, 5.5474, 5.2028, 7.6842, 18.2482, 14.399, 10.0604, 8.0892, 5.9807] 
        a = [8.46, 8.81, 8.52, 8.69, 7.76, 7.82, 10.15, 11.85, 11.67, 9.38, 8.73, 8.64] 
        k = [2.08, 2.736, 2.275, 2.143, 2.15, 1.791, 1.959, 2.529, 2.744, 2.49, 2.029, 1.971] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Thanet"

class Thor(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [2.7514, 3.4086, 6.0247, 8.1419, 6.5508, 6.0828, 7.7531, 11.9813, 12.794, 12.5341, 14.2796, 7.6978] 
        a = [7.58, 8.76, 11.21, 11.53, 12.32, 11.86, 12.21, 13.15, 12.95, 12.45, 12.92, 11.95] 
        k = [1.736, 2.432, 2.662, 2.307, 2.662, 2.264, 2.08, 2.279, 2.061, 2.158, 2.291, 2.076] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Thor"

class Thortonbank(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.5245, 7.2766, 7.0753, 7.1398, 5.2957, 4.3792, 8.3066, 12.2378, 18.9911, 10.0912, 7.4417, 6.2403] 
        a = [8.92, 8.86, 9.52, 10.13, 8.69, 8.95, 11.97, 12.18, 12.67, 10.24, 9.22, 9.16] 
        k = [2.369, 2.443, 2.654, 2.975, 2.521, 2.076, 2.482, 2.537, 2.467, 2.119, 1.955, 2.025] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Thortonbank"

class Thortonbank2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.5434, 7.2816, 7.019, 7.1075, 5.3642, 4.35, 8.2301, 12.27, 18.9486, 10.1094, 7.4551, 6.3212] 
        a = [8.93, 8.85, 9.51, 10.23, 8.67, 8.9, 11.95, 12.23, 12.61, 10.28, 9.26, 9.13] 
        k = [2.369, 2.42, 2.631, 3.119, 2.525, 2.053, 2.463, 2.529, 2.439, 2.127, 1.975, 2.014] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Thortonbank2"

class Trea_mollebugt(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [3.7682, 4.0183, 5.4401, 6.3481, 8.012, 8.6854, 9.0132, 11.8688, 13.7734, 15.9148, 9.4216, 3.7361] 
        a = [9.01, 8.37, 9.08, 10.39, 11.7, 11.28, 11.44, 13.06, 13.24, 13.77, 12.52, 9.61] 
        k = [2.135, 1.986, 2.049, 2.072, 2.271, 2.338, 2.197, 2.518, 2.381, 2.924, 2.166, 1.846] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Trea Mollebugt"

class Trianel_windpark(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.1019, 4.0394, 5.8772, 8.2256, 7.5374, 6.048, 7.4588, 12.0417, 14.4173, 12.2881, 9.6647, 8.2999] 
        a = [8.2, 7.92, 9.46, 10.9, 11.13, 10.55, 11.17, 13.3, 12.39, 11.99, 11.05, 9.81] 
        k = [2.029, 2.197, 2.232, 2.232, 2.818, 2.604, 2.096, 2.811, 2.404, 2.354, 2.475, 2.127] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Trianel Windpark"

class Triton_knoll(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [6.7195, 4.5885, 5.353, 6.3639, 5.6536, 7.6197, 9.0067, 15.1247, 14.3843, 11.3407, 8.2114, 5.634] 
        a = [9.81, 8.79, 8.7, 9.77, 9.35, 10.59, 11.19, 13.35, 12.77, 11.45, 10.33, 9.83] 
        k = [2.4, 2.428, 2.482, 2.205, 2.197, 2.51, 1.986, 2.494, 2.857, 2.357, 2.689, 2.104] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Triton Knoll"

class Tuno_knob(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [3.7088, 3.2094, 5.3767, 7.5389, 6.961, 7.8935, 8.2138, 14.2316, 12.2815, 12.8997, 12.1261, 5.559] 
        a = [7.62, 7.25, 8.49, 9.31, 9.57, 9.44, 8.56, 10.8, 11.21, 11.2, 11.22, 9.36] 
        k = [2.361, 2.471, 2.229, 2.287, 2.061, 2.279, 1.865, 2.584, 2.748, 2.58, 2.338, 2.416] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Tuno knob"

class Veja_mate(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.5357, 4.0318, 5.7623, 7.6063, 7.3531, 5.9795, 7.7502, 12.3357, 14.6211, 12.0716, 9.4088, 8.5441] 
        a = [8.76, 7.92, 8.91, 11.03, 11.3, 10.88, 11.24, 12.66, 12.54, 11.98, 11.35, 9.97] 
        k = [2.217, 2.236, 2.061, 2.283, 2.615, 2.592, 1.959, 2.424, 2.377, 2.307, 2.506, 2.115] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Veja Mate"

class Vestershav_nord(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [3.0839, 2.4023, 5.0598, 9.0082, 7.6797, 6.543, 7.3517, 12.1411, 12.8338, 12.5704, 13.9562, 7.3699] 
        a = [8.32, 9.21, 11.08, 11.82, 12.47, 12.17, 12.28, 13.19, 12.96, 12.32, 12.85, 11.29] 
        k = [1.9, 2.615, 2.564, 2.361, 2.654, 2.416, 2.162, 2.295, 2.076, 2.15, 2.35, 2.01] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Vestershav Nord"

class Vestershav_nord2(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [2.6189, 2.9633, 7.0073, 8.2579, 5.0963, 6.6959, 8.7822, 12.1812, 12.4499, 12.8975, 15.0696, 5.98] 
        a = [7.76, 8.92, 10.72, 11.63, 13.66, 13.39, 13.16, 13.01, 12.69, 12.34, 13.37, 11.64] 
        k = [2.061, 2.428, 2.287, 2.256, 2.814, 2.646, 2.451, 2.217, 2.014, 2.107, 2.365, 1.939] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Vestershav Nord2"

class Vindeby_fejo(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.1132, 2.6123, 5.3656, 8.8613, 6.4302, 6.4128, 8.848, 10.7662, 15.756, 16.1827, 9.2133, 5.4382]
        a = [7.1, 6.31, 8.37, 9.8, 10.49, 9.67, 9.88, 10.89, 11.42, 10.76, 9.84, 8.07]
        k = [1.842, 1.803, 2.1, 2.443, 2.619, 2.748, 2.561, 2.529, 2.428, 2.33, 2.33, 2.135]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Vindeby fejo"

class Waterkant(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.5425, 4.0467, 5.7492, 7.5938, 7.3171, 6.0053, 7.7457, 12.3593, 14.5959, 12.0658, 9.4258, 8.5528] 
        a = [9.09, 7.65, 8.92, 11.31, 11.75, 11.41, 11.61, 13.53, 13.53, 12.67, 11.9, 10.51] 
        k = [2.162, 2.045, 1.936, 2.033, 2.369, 2.365, 1.775, 2.311, 2.213, 2.201, 2.373, 2.115] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Waterkant"

class Westermost_Rough(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [7.2525, 5.3117, 5.1625, 6.0878, 5.1559, 8.1427, 8.736, 11.8812, 15.3404, 13.7191, 8.2592, 4.9509]
        a = [10.12, 8.6, 8.59, 9.25, 9.41, 10.54, 11.61, 13.02, 12.93, 12.18, 10.59, 9.98]
        k = [2.553, 2.408, 2.26, 2.146, 2.205, 2.604, 2.24, 2.607, 2.889, 2.674, 3.193, 2.346]
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Westermost Rough"

class Wind_energy_search_area_post_2030_3(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [5.3304, 4.6413, 5.5923, 6.9214, 5.8515, 5.7535, 9.0085, 13.6143, 14.5311, 11.7209, 9.5565, 7.4782] 
        a = [9.29, 7.58, 9.2, 10.9, 9.83, 10.05, 11.61, 14.01, 13.77, 12.8, 11.38, 10.38] 
        k = [1.986, 1.908, 2.135, 1.928, 1.967, 1.971, 1.748, 2.061, 2.299, 2.049, 2.318, 2.186] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Wind energy search area post 2030-3"

class Wind_energy_search_area_post_2030_8(UniformWeibullSite):
    def __init__(self, ti=0.07, shear=None):
        f = [4.7541, 4.6141, 6.368, 7.1474, 6.0256, 5.3254, 8.0244, 13.9878, 15.431, 11.4434, 9.4095, 7.4693] 
        a = [8.69, 7.91, 9.64, 11.04, 9.5, 9.78, 11.78, 13.88, 13.3, 12.61, 11.05, 10.07] 
        k = [1.967, 2.088, 2.162, 2.15, 1.971, 1.873, 1.865, 2.197, 2.189, 2.049, 2.186, 2.064] 
        UniformWeibullSite.__init__(self, np.array(f) / np.sum(f), a, k, ti=ti, shear=shear)
        self.initial_position = np.array([dummyWT_x, dummyWT_y]).T
        self.name = "Wind energy search area post 2030-8"

        #when TBA or not given, hub height = 180
