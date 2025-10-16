def get_aep4smart_start(windFarmModel, nCPU = 1, ws=[6, 8, 10, 12, 14], wd=np.arange(360), type=0, **kwargs):
        """Compute AEP with a smart start approach"""
        def aep4smart_start(X, Y, wt_x, wt_y, T=0, wt_t=0):
            H = np.full(X.shape, windFarmModel.windTurbines.hub_height())
            if type == 0:
                sim_res = windFarmModel(wt_x, wt_y, type=wt_t, wd=wd, ws=ws, n_cpu=nCPU, **kwargs)
                next_type = T
            else:
                type_ = np.atleast_1d(type)
                t = np.zeros_like(wt_x) + type_[:len(wt_x)]
                sim_res = windFarmModel(wt_x, wt_y, type=t, wd=wd, ws=ws, n_cpu=nCPU, **kwargs)
                H = np.full(X.shape, windFarmModel.windTurbines.hub_height())
                next_type = type_[min(len(type_) - 1, len(wt_x) + 1)]
            return sim_res.aep_map(Points(X, Y, H), type=next_type, n_cpu=nCPU).values
        return(aep4smart_start)