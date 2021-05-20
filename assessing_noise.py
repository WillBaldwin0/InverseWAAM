import numpy as np
from blocks.datatools.analysis import ExpData2

name = "H4_90_F2_random"
data_obj = ExpData2(name, fringe=10)

#%% zero strain datasets

rawdisps = data_obj.all_disps_dict
frames = data_obj.frames
avg_norm = 0

arr = []
for frame in frames:
    arr.append(rawdisps[frame][:, 1])
    print(np.std(rawdisps[frame][:, 1]))
    print(np.linalg.norm(rawdisps[frame][:, 1]))
    avg_norm += np.linalg.norm(rawdisps[frame][:, 1])
    print()
    
print('avg')
arr = np.asarray(arr).flatten()
stds = np.std(arr, axis=0)
print(stds)
print(avg_norm / len(frames))

