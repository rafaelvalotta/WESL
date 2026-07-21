import windIO

data = windIO.load_yaml("/Users/brunoboer/Documents/Software/Test_Wesl_jun23/wesl/Advanced_Optimizer/Data/vineyard_revolution_system.yaml")
windIO.validate(data, schema_type="plant/wind_energy_system")
