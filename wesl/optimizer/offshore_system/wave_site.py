import numpy as np
import xarray as xr
from scipy.stats import weibull_min



class wave_site(): 
    """
    Gets direction, height, period """
    def __init__(self, file_path, directions = 6):
        ds = xr.open_dataset(file_path)
        direction = ds["VMDR_SW1"].to_dataframe()["VMDR_SW1"].tolist()
        mean_period = ds["VTM01_SW1"].to_dataframe()["VTM01_SW1"].tolist()
        sig_height = ds["VHM0_SW1"].to_dataframe()["VHM0_SW1"].tolist()

        total_hours = len(direction)

        self.direction_num = directions

        raw_occurances = {} # dictionary keys are direction, period, sig_height raw_occurances
        for i in range(total_hours):
            
            rounded_dir = np.round(direction[i] / 360 * self.direction_num).astype(int) 
            rounded_dir = rounded_dir % self.direction_num
            rounded_period = np.round(mean_period[i])
            rounded_height = np.round(sig_height[i])
            key = (rounded_dir, rounded_period, rounded_height)

            #count how many each triple occurred

            if(key in raw_occurances):
                raw_occurances[key] += 1
            else:
                raw_occurances[key] = 1

        # divide by total hours to find probability of each occurence    
        for key, value in raw_occurances.items():
            raw_occurances[key] = value / total_hours

        

        self.triple_probabilities = dict(sorted(raw_occurances.items(), key=lambda item: item[1], reverse=True))
        
            # raw_occurances[direction[i] // 60].append(direction[i], mean_period[i], sig_height[i])
            
    #returns direction height probabilities
    def probabilities(self, height_cutoffs):
        probabilities = np.zeros((self.direction_num, len(height_cutoffs)-1))
        for key, value in self.triple_probabilities.items():
            height_index = 0
            for index, height in enumerate(height_cutoffs):
                height_index = index
                # height index corresponds to one of the ranges between cutoff values
                if key[2] > height:
                    continue
                else:
                    break

            probabilities[int(key[0]),height_index] += value
            
        return np.array(probabilities)
    
    # returns direction period height probabilities
    def probability_triples(self, period_cutoffs, height_cutoffs):
        probabilities = np.zeros((self.direction_num, len(period_cutoffs)-1, len(height_cutoffs)-1))
        for key, value in self.triple_probabilities.items():
            height_index = 0
            period_index = 0
            for index, height in enumerate(height_cutoffs):
                height_index = int(index)
                # height index corresponds to one of the ranges between cutoff values
                if key[2] > height:
                    continue
                else:
                    break
            for index, period in enumerate(period_cutoffs):
                period_index = int(index)
                if key[1] > period:
                    continue
                else:
                    break
            probabilities[int(key[0]), period_index, height_index] += value

        return np.array(probabilities)

class wave_site_weibull():
    def __init__(self, f, a, k, period_a, period_k):
        if (len(f) != len(a) or len(f) != len(k)):
            raise ValueError("weibull parameters for waves are not equal")

        self.f = f
        self.a = a
        self.k = k
        self.period_a = period_a
        self.period_k = period_k
        self.weibull = []
        # for i in range(len(self.f)):
        #     self.weibull.append(weibull_min(c=self.k[i], scale = a))

    def probabilities(self, buckets):

        # probability matrix of the form [directions, wave_height]
        probabilities = []
        for i in range(len(self.f)):
            cdf_values = weibull_min.cdf(buckets, c=self.k[i], scale = self.a[i])
            bucket_prob = np.diff(cdf_values)
            probabilities.append(bucket_prob*self.f[i])
        return np.array(probabilities)

    def probability_triples(self, bucket_period, bucket_height):
        probabilities = []
        for i in range(len(self.f)):
            height_period_prob_matrix = []
            cdf_values = weibull_min.cdf(bucket_height, c=self.k[i], scale = self.a[i])
            bucket_prob_height = np.diff(cdf_values)
            for j in range(len(bucket_prob_height)):
                    cdf_values = weibull_min.cdf(bucket_period, c=self.period_k[i], scale = self.period_a[i])
                    bucket_prob_period = np.diff(cdf_values)
                    height_period_prob_matrix.append(bucket_prob_period*bucket_prob_height[j])
            height_period_prob_matrix = np.array(height_period_prob_matrix)
            probabilities.append(height_period_prob_matrix*self.f[i])
        return np.array(probabilities)