import os
import cooler
import shutil
from main_params import MainParams

def check_resolutions(mcool_file):
    resolutions = []
    with cooler.Cooler(mcool_file) as c:
        for group in c.groups:
            if group.startswith('resolutions/'):
                resolutions.append(int(group.split('/')[-1]))

    return resolutions

p = MainParams()
sp = "hg38"
folder_path = f"/media/dl-box/T7/{sp}_hic"
file_list = os.listdir(folder_path)

hic_list = []
for filename in file_list:
    try:
        file_path = os.path.join(folder_path, filename)
        c = cooler.Cooler(file_path + "::resolutions/" + str(p.hic_bin_size))
        c.matrix(balance=True, field="count").fetch(f'chrX:100000-200000')[0, -1]
        if filename in p.hic_keys[sp].to_list():
            print(filename)
            hic_list.append(filename)
            shutil.copy(file_path, p.hic_folder + filename)
    except:
        pass
print(len(hic_list))
with open(f"{sp}_hic.tsv", 'w') as file:
    file.write('\n'.join(hic_list))
