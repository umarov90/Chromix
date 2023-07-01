from main_params import MainParams
import pickle5 as pickle
import joblib

p = MainParams()

def convert(path):
  with open(path, "rb") as fh:
    data = pickle.load(fh)
    joblib.dump(data, path)


convert(p.model_path + "_res")
convert(p.model_path + "_expression")
convert(p.model_path + "_epigenome")
convert(p.model_path + "_conservation")
convert(p.model_path + "_hic")