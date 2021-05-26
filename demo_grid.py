from blocks.fem.simple_2d_newps import Experiment_ortho2d_extra
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from blocks.datatools.analysis import ExpData2, gradient_vector_svd
    
#%% dataobj

name = "H4_90_F1"
data_obj = ExpData2(name, fringe=20)
gvec = gradient_vector_svd(data_obj.centered_trunc_disps)
# gvec needs correct strain
coords = data_obj.centered_trunc_coords
slope, intr, _,_,_ = linregress(coords[:,1], gvec)
disps = - gvec * 10
strain = - slope * 10

#%% make the model

# 2,9. 3,17. 4,25. 5,33. 
nx = 3
ny = 17
n = nx*ny
Exp = Experiment_ortho2d_extra(data_obj, nx, ny, method='linear')


#%%

params_shape = Exp.model_matrix.shape[1]

rand_params = np.random.rand(params_shape)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18

Exp.contour_b_vec(Exp.model_matrix.dot(rand_params), 0, fringe=True, orientation='vertical')