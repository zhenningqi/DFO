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

os.environ["PYCUTEST_CACHE"] = "/Users/qizhenning/Documents/DFO_solver/pycutest_cache"
os.environ["MACOSX_DEPLOYMENT_TARGET"] = "14.0"

import pycutest


def get_prob(prob_name):
    prob = pycutest.import_problem(prob_name)

    dim = prob.n
    x0 = prob.x0
    obj_func = prob.obj

    return {"prob_name": prob_name, "dim": dim, "x0": x0, "obj_func": obj_func}


# prob_name = "SISSER"
# prob_dict = get_prob(prob_name)
# print(prob_dict)
