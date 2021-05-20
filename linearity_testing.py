from blocks.fem.simple_2d_newps import Experiment_ortho2d_extra
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from blocks.datatools.analysis import ExpData2, gradient_vector_svd
    
#%% make data object and get the relevant measurement

name = "H4_90_F1"
data_obj = ExpData2(name, fringe=20)
gvec = gradient_vector_svd(data_obj.centered_trunc_disps)

# gvec needs correct strain
coords = data_obj.centered_trunc_coords
slope, intr, _,_,_ = linregress(coords[:,1], gvec)
disps = - gvec * 10
strain = - slope * 10

#%% make the experiment object and set meaurement scheme and model
nx = 3
ny = 17
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
r=10
inverted_b, d_coefs, U_red, svals, p_coefs, V_red = Exp.invert_measurement(r, strain, disps)

# construct the expected dispalcement at this point, which minimises cost in the tangent space
linear_d = m0 + U_red.dot(d_coefs)
m1 = Exp.get_measurement(strain, inverted_b)

print(np.linalg.norm(m1-disps))
print(np.linalg.norm(m1-linear_d))

print(np.std(d_coefs[30:]))

#%% plot displacement coefs and noise level
plt.figure()
plt.plot(d_coefs)
sigma = 0.002
plt.plot(np.ones(d_coefs.shape)*sigma, '--')
plt.plot(-np.ones(d_coefs.shape)*sigma, '--')
plt.show()

#%% draw reconstruction
cbar = Exp.contour_b_vec(inverted_b, 1, fringe=False, title='reconstruction')

#%% show 1d linear approx
Exp.scatter_measurement_1d(m1-m0, linear_d-m0, 1)

#%% compute the reconstruction using an arbitrary set

r=51
inverted_b, d_coefs, U_red, svals, p_coefs, V_red = Exp.invert_measurement(r, strain, disps)

errs = []
fracerrs = []

pred_costs = []
costs = []

for i in range(41):
    include_vecs = [j+i for j in range(10)]
    
    inverted_b, d_coefs, U_red, svals, p_coefs, V_red
    
    lin_pred_d = m0 + U_red[:, include_vecs].dot(d_coefs[include_vecs])
    lin_extrap_b = Exp.default_b_vector + Exp.model_matrix.dot(V_red[:, include_vecs].dot(p_coefs[include_vecs]))    
    here_actual_d = Exp.get_measurement(strain, lin_extrap_b)
    
    pred_costs.append(np.linalg.norm(disps - lin_pred_d))    
    costs.append(np.linalg.norm(disps - here_actual_d))
    
    errs.append(np.linalg.norm(here_actual_d - lin_pred_d))
    fracerrs.append(np.linalg.norm(here_actual_d - lin_pred_d) / np.linalg.norm(lin_pred_d - m0))

#%%
plt.figure()
plt.plot(10 + np.array(range(len(errs))), np.log10(errs))
plt.xlabel('rank')
plt.ylabel('log norm difference')
plt.xlim([0,51])
plt.show()

plt.figure()
plt.plot(10 + np.array(range(len(errs))), np.log10(fracerrs))
plt.xlabel('rank')
plt.ylabel('log norm fractional difference')
plt.xlim([0,51])
plt.show()

#%%

print(np.linalg.norm(disps-m0))

#%% compute the reconstruction using an arbitrary set

r=51
inverted_b, d_coefs, U_red, svals, p_coefs, V_red = Exp.invert_measurement(r, strain, disps)

errs = []
include_vecs = []
pred_costs = []
costs = []

for i in range(51):
    include_vecs.append(i)

    inverted_b, d_coefs, U_red, svals, p_coefs, V_red
    
    lin_pred_d = m0 + U_red[:, include_vecs].dot(d_coefs[include_vecs])
    lin_extrap_b = Exp.default_b_vector + Exp.model_matrix.dot(V_red[:, include_vecs].dot(p_coefs[include_vecs]))
    here_actual_d = Exp.get_measurement(strain, lin_extrap_b)
    
    pred_costs.append(np.linalg.norm(disps - lin_pred_d))    
    costs.append(np.linalg.norm(disps - here_actual_d))
    
    errs.append(np.linalg.norm(here_actual_d - lin_pred_d))

#%%
plt.figure()
plt.plot(errs)
plt.show()

#%%
plt.figure()
plt.plot(costs, label='actual cost at solution')
plt.plot(pred_costs, label='anticipated cost at solution')
plt.xlabel('rank')
plt.ylabel('cost')
plt.legend()
plt.show()

#%%
plt.figure()
plt.plot(np.log10(costs))
plt.plot(np.log10(pred_costs))
plt.show()

#%%
plt.figure()
plt.plot(np.log10(errs))
plt.show()

#%%

print(np.linalg.norm(disps-m0))