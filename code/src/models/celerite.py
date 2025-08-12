from src.models.base_model import BaseModel

import numpy as np;

import celerite2;
from celerite2 import terms;

from scipy.optimize import minimize;

class Celerite(BaseModel):
    def __init__(self):
        super().__init__("Celerite");
        self.__trained = False;

        self.__length_scale = 1.0;

        self.__kernel = terms.Matern32Term(sigma = 1.0, rho = self.__length_scale);
        self.__model = celerite2.GaussianProcess(self.__kernel);

        # Define fixed machine epsilon jitter
        self.__yerr = float(1e-8);

    def train(self, data_X, data_y):
        self.__trained = True;
        
        data_X, data_y = self.__one_d_data(data_X, data_y); 

        self.__model.compute(data_X, yerr = self.__yerr);
        print("Initial log-likelihood: ", self.__model.log_likelihood(data_y));
        
        # Optimise mean and sigma_s only
        init_mean = 0;
        init_sigma = 1;

        init_params = [init_mean, np.log(init_sigma)];
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
        self.__model.mean = params[0];
        log_sigma = params[1];

        sigma = np.exp(log_sigma);

        self.__kernel = terms.Matern32Term(sigma = sigma, rho = self.__length_scale);
        self.__model.compute(data_X, quiet = True);

    def predict(self, test_X, train_y):
        if self.__trained == False:
            raise Exception("Model not trained yet");

        test_X, train_y = self.__one_d_data(test_X, train_y); 

        mu, var = self.__model.predict(train_y, t = test_X, return_var = True);
        return mu, var;
