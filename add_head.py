import joblib
import model as mo
from main_params import MainParams
from torch import nn
import parse_data as parser
import torch

p = MainParams()
train_info, valid_info, test_info = parser.parse_sequences(p)
heads = joblib.load(f"{p.pickle_folder}heads.gz")
heads["hg38"]["expression_sc"] = parser.parse_tracks_sc(p, train_info + valid_info + test_info)
joblib.dump(heads, f"{p.pickle_folder}heads.gz", compress="lz4")

model, _ = mo.prepare_model(p)
start_epoch = mo.load_weights(p, model)
model.heads["hg38_expression_sc"] = nn.Sequential(
    model.crop_final,
    nn.Linear(4 * p.dim, len(heads["hg38"]["expression_sc"]))
)

torch.save({'epoch': 0, 'model_state_dict': model.state_dict()}, p.model_folder + p.model_name)
