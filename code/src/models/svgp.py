from src.models.base_model import BaseModel

import numpy as np;

import GPy;
import climin;

# Monkey patches for climin that hasn't updated to python 3.10 yet

# Iterables
import collections;
import collections.abc;
collections.Iterable = collections.abc.Iterable;

# Mutable ranges
import climin.util as cu
def _draw_mini_slices_py3(n_samples, batch_size):
    import random
    idxs = list(range(n_samples)) 
    while True:
        random.shuffle(idxs)
        for start in range(0, n_samples, batch_size):
            yield idxs[start:start + batch_size]
cu.draw_mini_slices = _draw_mini_slices_py3


class SVGP(BaseModel):
    def __init__(self):
        super().__init__("SVGP");

        # Use RBF kernel
        self.__kernel = GPy.kern.RBF(input_dim=1);

        self.__kernel.lengthscale.constrain_bounded(0.08, 0.20);

        self.__likelihood = GPy.likelihoods.Gaussian();
        
    def train(self, data_X, data_y):
        self.__trained = True;
        
        n_X = data_X.shape[0];
        
        # Normalise the data
        self.__xmin = data_X.min();
        self.__xscale = data_X.max() - data_X.min(); 
        norm_X = (data_X - self.__xmin) / self.__xscale;

        self.__ymean = data_y.mean();
        self.__yscale = float(data_y.std() + 1e-12);
        norm_y = (data_y - self.__ymean) / self.__yscale;

        # Initialise likelihood variance from data
        hf = np.diff(norm_y.ravel());
        sig = 1.4826 * np.median(np.abs(hf)) / np.sqrt(2.0);
        self.__likelihood.variance = sig ** 2;
        self.__likelihood.variance.constrain_bounded(1e-6, 0.1);
        
        self.__kernel.variance.constrain_bounded(0.1, 2.0);

        # Create inducing points at random
        m = max(256, min(512, n_X // 10));
        M = np.linspace(
            0,
            1,
            m,
        dtype = float).reshape(-1, 1);
        
        # Enable minibatching
        batch_size = min(1024, n_X);
        
        self.__model = GPy.core.SVGP(
            norm_X,
            norm_y,
            M,
            self.__kernel,
            self.__likelihood,
            batchsize = batch_size,
        );

        # Use Adadelta optimiser
        self.__optimiser = climin.Adam(
            self.__model.optimizer_array,
            self.__model.stochastic_grad,
            step_rate = 1e-3,
        );

        info = self.__optimiser.minimize_until(self.__optimiser_callback);

    def __optimiser_callback(self, info):
        iteration = info.get("n_iter", info.get("iteration", 0));
        
        if iteration >= self._max_iterations:
            print(f"Reached maximum iterations: {iteration}");
            return True;
        
        # Reduce overhead
        if iteration % self._eval_every != 0:
            return False;
         
        ll = float(self.__model.log_likelihood());
        
        # Initialise if first measurement
        if not hasattr(self, '_SVGP__prev_ll'):
            self.__prev_ll = ll;
            self._stall_count = 0;
            print(f"Iteration {iteration}: Log likelihood = {ll} (initial)");
            return False; 

        # Improvement check
        improvement = ll - self.__prev_ll;

        if improvement < self._tolerance:
            self._stall_count += 1;
            print(f"Iteration {iteration}: Log likelihood = {ll} (stall, stall count: {self._stall_count})");
            
            if self._stall_count >= self._max_stalls:
                print(f"Reached maximum stalls: {self._stall_count}");
                return True;

        else:
            ();
            # self._stall_count = 0;
            print(f"Iteration {iteration}: Log likelihood = {ll} (improvement: {improvement})");

        self.__prev_ll = ll;
        return False;

    def predict(self, data_X, data_y):
        if self.__trained == False:
            raise Exception("Model has not been trained yet.");
        
        data_X = np.asarray(data_X, dtype=float).reshape(-1, 1);

        # Normalise the data
        norm_X = (data_X - self.__xmin) / self.__xscale;
        mu, variance = self.__model.predict(norm_X);
        mu = mu * self.__yscale + self.__ymean;
        variance = variance * self.__yscale ** 2;

        return mu, variance;
