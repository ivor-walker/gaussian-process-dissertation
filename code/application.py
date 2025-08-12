import src.data_processing as data_processing;

import src.data_viz as data_viz;

from src.models.celerite import Celerite;
from src.models.svgp import SVGP;

# Set seeds
import numpy as np;
np.random.seed(42);

import random;
random.seed(42);

data, data_X, data_y = data_processing.process("data/spSpec-51613-0305-552_resid_noGP.csv");
data_gp, data_gp_X, data_gp_y = data_processing.process("data/spSpec-51613-0305-552_resid_GP.csv");

models = [
    Celerite(),
    SVGP(),
];

training_times = [model.time_train(data_X, data_y) for model in models];
print(training_times);

predict_times = [model.time_predict(data_X, data_y) for model in models];
print(predict_times);

data_viz.svgp_vs_celerite(predict_times, data_X, data_y);
