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

import numpy as np


def rosenbrock(x):
    a = 1.0
    b = 100.0
    return np.sum(b * (x[1:] - x[:-1] ** 2) ** 2 + (a - x[:-1]) ** 2)


def get_easy_prob(dim):
    prob_name = f"{dim}dim_Rosenbrock"

    x0 = np.zeros(dim)
    obj_func = rosenbrock

    return {"prob_name": prob_name, "dim": dim, "x0": x0, "obj_func": obj_func}


# dim = 10
# prob_dict = get_easy_prob(dim)
# print(prob_dict)
