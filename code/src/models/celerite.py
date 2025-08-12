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
        
        # Define bounds on optimiser, especially to keep length scale long
        self.__lower_length_scale = 500;

        
    def train(self, data_X, data_y, yerr = 0.8):
        self.__trained = True;
        
        data_X, data_y = self.__one_d_data(data_X, data_y); 
        
        n_X = len(data_X);
        self.__yerr = np.full(n_X, float(yerr), dtype=float);
        
        # Initial guesses based on data
        mean_guess = float(np.median(data_y));
        sigma_guess = float(np.std(data_y));
        length_scale_guess = float(data_X.max() - data_X.min()) * 0.3;

        self.__kernel = terms.Matern32Term(
            sigma = sigma_guess,
            rho = length_scale_guess
        );
        self.__model = celerite2.GaussianProcess(self.__kernel, mean = mean_guess);
        self.__model.compute(data_X, yerr = self.__yerr);

        print("Initial log-likelihood: ", self.__model.log_likelihood(data_y));
        
        # Create bounds based on initial guesses, especially to keep length scale long
        self.__optimiser_bounds = {
            "mean": (None, None),
            "sigma": (np.log(1e-6 * sigma_guess + 1e-12), np.log(1e6 * sigma_guess + 1e-12)),
            "rho": (np.log(max(self.__lower_length_scale, length_scale_guess * 0.05)), 
                    np.log(length_scale_guess * 20)),
        };
        self.__optimiser_bounds = self.__optimiser_bounds.values();

        init_params = [
            mean_guess,
            np.log(sigma_guess + 1e-12),
            np.log(length_scale_guess),
        ];

        solution = minimize(
            self.__neg_log_like,
            init_params,
            args = (data_X, data_y),
            method = 'L-BFGS-B',
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

        self.__model.mean = mean;
        self.__kernel = terms.Matern32Term(sigma = log_sigma, rho = log_length_scale);

        self.__model.kernel = self.__kernel;
        self.__model.compute(data_X, yerr = self.__yerr, quiet = True);

    def predict(self, test_X, train_y):
        if self.__trained == False:
            raise Exception("Model not trained yet");

        test_X, train_y = self.__one_d_data(test_X, train_y); 

        mu, var = self.__model.predict(train_y, t = test_X, return_var = True);
        return mu, var;
