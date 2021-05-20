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
name = "H1_90_F1"
data_obj = ExpData2(name, fringe=20)
gvec = gradient_vector_svd(data_obj.centered_trunc_disps)

#%%

#gvec = 0.1*data_obj.centered_trunc_disps[data_obj.frames[2]][:, 1]

# gvec needs correct strain
coords = data_obj.centered_trunc_coords
slope, intr, _,_,_ = linregress(coords[:,1], gvec)
disps = - gvec * 10
strain = - slope * 10

#%% make the experiment object and set meaurement scheme and model
nx = 5
ny = 33
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

#%% invert using first r singular vectors
r=45
inverted_b, d_coefs, U_red, svals, p_coefs, V_red = Exp.invert_measurement(r, strain, disps)

# construct the expected dispalcement at this point, which minimises cost in the tangent space
linear_d = m0 + U_red.dot(d_coefs)
m1 = Exp.get_measurement(strain, inverted_b)

print(np.linalg.norm(m1-disps))
print(np.linalg.norm(m1-linear_d))
#print(np.std(d_coefs[30:]))

#%% plot displacement coefs and noise level
plt.figure()
plt.plot(d_coefs)
sigma = 0.003
#plt.plot(np.ones(d_coefs.shape)*sigma, '--')
#plt.plot(-np.ones(d_coefs.shape)*sigma, '--')
plt.show()

#%% plot s-vals
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.plot(np.log10(np.reciprocal(np.asarray(svals))))
plt.xlabel('singular vector rank')
plt.ylabel('- log s_i')
plt.show()

#%%
# Exp.contour_b_vec(inverted_b, 1, fringe=False, cbar=cbar, title='reconstruction')

#%% draw reconstruction
cbar = Exp.contour_b_vec(inverted_b, 1, fringe=False, title='reconstruction')

#%% show 1d linear approx
Exp.scatter_measurement_1d(m1-m0, linear_d-m0, 1)

#%% show 1d 
Exp.scatter_measurement_1d(m1-m0, disps-m0, 1)

#%% plot some s-vecs on unified scales
# from the back
cbar = Exp.contour_b_vec(Exp.params_to_b(Exp.param_s_vecs[:, -1]), 1, fringe=True, title='SV ' + str(n-1))
for i in range(1):
    Exp.contour_b_vec(Exp.params_to_b(Exp.param_s_vecs[:, -i-2]-0.05), 1, cbar=cbar, fringe=False, title='SV ' + str(n-i-2))
    
#%% plot some s-vecs
# from teh front
cbar = Exp.contour_b_vec(Exp.params_to_b(Exp.param_s_vecs[:, 0]), 1, fringe=True, title='SV ' + str(0))
for i in range(4):
    Exp.contour_b_vec(Exp.params_to_b(Exp.param_s_vecs[:, 1+i]), 1, fringe=True, title='SV ' + str(1+i))
    
#%% plot some d-space s-vecs on unified scales
# from the back
cbar = Exp.scatter_measurement(Exp.measurement_s_vecs[:, 0], 1)
for i in range(4):
    Exp.scatter_measurement(Exp.measurement_s_vecs[:, i+1], 1)
    
#%% compare predicted displacement and measured one, and plot error plot

cbar = Exp.scatter_measurement(disps-m0, 1)
Exp.scatter_measurement(m1-m0, 1, cbar=cbar)

#%%
Exp.scatter_measurement(np.abs(disps-m1), 1, cbar=cbar)

#%%
Exp.scatter_measurement(disps, 1)

#%% plot error field
pvecs = Exp.param_s_vecs.copy()
svals_inverse = np.reciprocal(np.asarray(Exp.s_vals))

for ind in range(pvecs.shape[1]):
    pvecs[:, ind] *= svals_inverse[ind]*sigma

squared = np.square(pvecs)

squared = squared[:, :13]
summed = np.sum(squared, axis=1)

rooted = np.sqrt(summed)
#cbar2 = Exp.contour_b_vec(Exp.params_to_b(rooted), 1)
Exp.contour_b_vec(Exp.params_to_b(rooted), 1, keep_boundaries=0, cbar=cbar2, orientation='horizontal')