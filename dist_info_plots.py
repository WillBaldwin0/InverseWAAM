from blocks.fem.simple_2d_newps import Experiment_ortho2d_extra
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from blocks.datatools.analysis import ExpData2, gradient_vector_svd
from matplotlib.ticker import MaxNLocator

""" infers Ey from y-displacement only, using a model in which Ex, nu_xy, and G are fixed. 
    inspection of the singular vectors when this model is not fixed shows that the other parameters 
    have small components compared to Ey in this set up """
    
#%% make data object and get the relevant measurement
name = "H4_90_F2"
data_obj = ExpData2(name, fringe=20)
gvec = gradient_vector_svd(data_obj.centered_trunc_disps)

#%%
# gvec needs correct strain
coords = data_obj.centered_trunc_coords
slope, intr, _,_,_ = linregress(coords[:,1], gvec)
disps = - gvec * 10
strain = - slope * 10


#%% make the experiment object and set meaurement scheme and model
nx = 4
ny = 25
n = nx*ny
Exp = Experiment_ortho2d_extra(data_obj, nx, ny, method='linear')

# set measurement and model - E_y only and d_y only 
Exp.set_special_measurement('all_y')
mat = np.zeros((n*4, n))
eye10 = np.eye(n)
mat[n:2*n] = eye10
Exp.set_model_matrix(Exp.double_sides_metric().dot(mat))

#%% derivative
m0 = Exp.compute_svd_at_b0(Exp.default_b_vector, strain)

#%%

y_i = Exp.measurement_s_vecs[:,:100].transpose().dot(disps-m0)

#%% plotting

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14

plt.figure()
plt.plot(np.abs(y_i))
plt.xlim([0, len(y_i)])
plt.ylim([0, max(np.abs(y_i))*1.2])
plt.xlabel('singular vector rank', labelpad = 10)
plt.ylabel('displacement coefficient y_i', labelpad = 15)
plt.show()