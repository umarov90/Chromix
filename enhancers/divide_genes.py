from main_params import MainParams
import joblib
import parse_data as parser

p = MainParams()
train_info, valid_info, test_info = parser.parse_sequences(p)
infos = train_info + valid_info + test_info

fractions = [0.3, 0.3, 0.1, 0.2]
sizes = [int(len(infos)*f) for f in fractions]
sizes[-1] += len(infos) - sum(sizes)

start = 0
for i, size in enumerate(sizes):
    end = start + size
    joblib.dump(infos[start:end], str(i) + "_" + str(size) + "_infos.p")
    start = end