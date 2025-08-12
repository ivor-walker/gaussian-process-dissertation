from src.models.base_model import BaseModel

import numpy as np;

import celerite2;
from celerite2 import terms;

from scipy.optimize import minimize;

class Celerite(BaseModel):
    def __init__(self):
        super().__init__("Celerite");
        self.__trained = False;

        self.__yerr = None;
        
        # Define lower length scale bound to avoid capturing high-frequency noise 
        # Measured in fraction of span, i.e. length scale scaled by the range of data
        self.__lower_length_scale = 0.10;
        self.__upper_length_scale = 3;

        
    def train(self, data_X, data_y, yerr_min = 0.8, yerr_scale = 1.4826, yerr_boost = 2):
        self.__trained = True;
        
        data_X, data_y = self.__one_d_data(data_X, data_y); 
        
        n_X = len(data_X);

        # Estimate model parameters using centred y
        centered_y = data_y - np.median(data_y); 
        mean_guess = 0 
        sigma_guess = float(np.std(centered_y));

        # Estimate yerr using centred y also
        centered_diff_y = np.diff(centered_y);
        yerr_est = yerr_scale * np.median(np.abs(centered_diff_y)) / np.sqrt(2.0);
        self.__yerr = np.full(n_X, float(max(yerr_est, yerr_min)), dtype = float) * yerr_boost;
        
        self.__length_scale_geometry = float(data_X.max() - data_X.min());
        length_scale_guess = self.__length_scale_geometry * self.__lower_length_scale 

        self.__kernel = terms.Matern32Term(
            sigma = sigma_guess,
            rho = length_scale_guess 
        );
        self.__model = celerite2.GaussianProcess(self.__kernel, mean = mean_guess);
        self.__model.compute(data_X, yerr = self.__yerr);

        print("Initial log-likelihood: ", self.__model.log_likelihood(data_y));
        
        # Create bounds based on initial guesses, especially to keep length scale long, in terms of fractions
        self.__optimiser_bounds = [
            (None, None),
            (np.log(0.5 * sigma_guess), np.log(2.0 * sigma_guess)),
            (np.log(self.__lower_length_scale), (np.log(self.__upper_length_scale))) 
        ];

        init_params = [
            mean_guess,
            np.log(sigma_guess),
            np.log(self.__lower_length_scale)
        ];
        
        print(f"Optimising with initial parameters: {init_params} and bounds: {self.__optimiser_bounds}");

        solution = minimize(
            self.__neg_log_like,
            init_params,
            args = (data_X, data_y),
            method = 'L-BFGS-B',
            bounds = self.__optimiser_bounds,
        );

        self.__set_params(solution.x, data_X);
    
    def __one_d_data(self, data_X, data_y):
        # Force data to be 1D
        data_X = np.asarray(data_X).reshape(-1);
        data_y = np.asarray(data_y).reshape(-1);

        return data_X, data_y;
    
    def __neg_log_like(self, params, data_X, data_y):
        self.__set_params(params, data_X);
        return -self.__model.log_likelihood(data_y);

    def __set_params(self, params, data_X):
        mean, log_sigma, log_length_scale = params;
        
        # De-log sigma and length scale
        sigma = float(np.exp(log_sigma));
        rho_frac = float(np.exp(log_length_scale));
        length_scale = self.__length_scale_geometry * rho_frac;
        
        print(f"Given parameters {params}, setting parameters: mean={mean}, sigma={sigma}, length scale frac={rho_frac}, length_scale={length_scale}, yerr={self.__yerr[0]}");
        self.__model.mean = mean;
        self.__kernel = terms.Matern32Term(sigma = sigma, rho = length_scale);

        self.__model.kernel = self.__kernel;
        self.__model.compute(data_X, yerr = self.__yerr, quiet = True);

    def predict(self, test_X, train_y):
        if self.__trained == False:
            raise Exception("Model not trained yet");

        test_X, train_y = self.__one_d_data(test_X, train_y); 

        mu, var = self.__model.predict(train_y, t = test_X, return_var = True);
        return mu, var;
