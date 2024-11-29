import sys
import os

"""
When running the code under the project root directory, getcwd will return the project root directory no matter what directory the code file is in.
However, python will set the directory where the code file is to default path, the project root directory will not be included in the default path.
Thus, if we want to use import, we should first set the project root directory into default path.
However, it is not true in notebook. In notebook, getcwd will return the folder where the file is.
"""
current_path = os.getcwd()
sys.path.append(current_path)

# os.environ["PYCUTEST_CACHE"] = "/home/qizhenning/qzn/DFO_solver/pycutest_cache"
os.environ["PYCUTEST_CACHE"] = "/Users/qizhenning/Documents/DFO_solver/pycutest_cache"
os.environ["MACOSX_DEPLOYMENT_TARGET"] = "14.0"

import random
import numpy as np

from CutestProb.GetProb import get_prob
from EasyProb.GetEasyProb import get_easy_prob


# set random seed
def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


set_random_seed(42)

#########################################################################
# Cutest prob
prob_name = "ERRINRSM"
# prob_name = "LUKSAN21LS"
# prob_name = "BROWNAL"
prob_dict = get_prob(prob_name)

# Rosenbrock
# dim = 100
# prob_dict = get_easy_prob(dim)
#########################################################################

x0 = prob_dict["x0"]
obj_func = prob_dict["obj_func"]

from Subspace import Solver

# use subspace solver to solve the easy problem
# create solver
solver = Solver.solver(obj_func, x0)

# initialize solver
max_iter = 1e8
max_nfev = 5e4

solver.init_option(
    max_iter,
    max_nfev,
    mom_num=1,
    gd_num=1,
    ds_num=0,
    extra_sample_num=0,
    gd_estimator_opt="FFD",
    gd_sample_num_opt=1,
)
solver.init_hyperparams()

# solve problem by the solver
# solver.solve()
solver.observe(landscape_freq=1)

# show result
solver.display_result()
solver.draw_info()
