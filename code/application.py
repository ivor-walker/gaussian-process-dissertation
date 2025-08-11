import src.data_processing as data_processing;

import src.data_viz as data_viz;

from src.models.celerite import Celerite;
from src.models.svgp import SVGP;

data = data_processing.process("data/spSpec-51613-0305-552_resid_noGP.csv");
data_gp = data_processing.process("data/spSpec-51613-0305-552_resid_GP.csv");

models = [
    Celerite(),
    SVGP(),
];

training_times = [model.time_train(data) for model in models];
predict_times = [model.time_predict(data) for model in models];

breakpoint();
