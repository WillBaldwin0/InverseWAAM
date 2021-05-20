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
name = "H4_90_F1"
data_obj = ExpData2(name, fringe=20)
gvec = gradient_vector_svd(data_obj.centered_trunc_disps)

#%%
# gvec needs correct strain
coords = data_obj.centered_trunc_coords
slope, intr, _,_,_ = linregress(coords[:,1], gvec)
disps = - gvec * 10
strain = - slope * 10

#%%
ranks = {'20':16, '10':30, '6.7':50, '5':50}
cutoffs = {'20':16, '10':43, '6.7':81, '5':131}
nxs = {'20':2, '10':3, '6.7':4, '5':5}
nys = {'20':9, '10':17, '6.7':25, '5':33}

#%% make the experiment object and set meaurement scheme and model

mod_name = '6.7'

nx = nxs[mod_name]
ny = nys[mod_name]
n = nx*ny
Exp = Experiment_ortho2d_extra(data_obj, nx, ny, method='linear')

# set measurement and model - E_y only and d_y only 
Exp.set_special_measurement('all_y')
mat = np.zeros((n*4, n))
eye10 = np.eye(n)
mat[n:2*n] = eye10
Exp.set_model_matrix(Exp.double_sides_metric().dot(mat))

m0 = Exp.compute_svd_at_b0(Exp.default_b_vector, strain)
#%%
r = 100#ranks[mod_name]

inverted_b, d_coefs, U_red, svals, p_coefs, V_red = Exp.invert_measurement(r, strain, disps)

# construct the expected dispalcement at this point, which minimises cost in the tangent space
linear_d = m0 + U_red.dot(d_coefs)
m1 = Exp.get_measurement(strain, inverted_b)

print(np.linalg.norm(m1-disps))
print(np.linalg.norm(m1-linear_d))

cbar = Exp.contour_b_vec(inverted_b, 1, fringe=False, title='reconstruction')

# plot error field
sigma = 0.003

pvecs = Exp.param_s_vecs.copy()
svals_inverse = np.reciprocal(np.asarray(Exp.s_vals))

for ind in range(pvecs.shape[1]):
    pvecs[:, ind] *= svals_inverse[ind]*sigma

squared = np.square(pvecs)


cutoff = cutoffs[mod_name]

squared = squared[:, :cutoff]
summed = np.sum(squared, axis=1)

rooted = np.sqrt(summed)
cbar2 = Exp.contour_b_vec(Exp.params_to_b(rooted), 1, cmap='Oranges')
#Exp.contour_b_vec(Exp.params_to_b(rooted), 1, keep_boundaries=0, cbar=cbar2, orientation='horizontal')

#%%

#
#
#
plt.figure()
plt.plot(np.abs(d_coefs))
plt.xlabel('singular vector rank')
plt.ylabel('displacement singular vector coeficient y_i')
plt.show()

##
#
##
##

#%% make the experiment object and set meaurement scheme and model

for mod_name in ['20','10','6.7']: # ,'6.7','5'
    
    nx = nxs[mod_name]
    ny = nys[mod_name]
    n = nx*ny
    Exp = Experiment_ortho2d_extra(data_obj, nx, ny, method='linear')
    
    # set measurement and model - E_y only and d_y only 
    Exp.set_special_measurement('all_y')
    mat = np.zeros((n*4, n))
    eye10 = np.eye(n)
    mat[n:2*n] = eye10
    Exp.set_model_matrix(Exp.double_sides_metric().dot(mat))
    
    m0 = Exp.compute_svd_at_b0(Exp.default_b_vector, strain)
    
    r = ranks[mod_name]
    
    inverted_b, d_coefs, U_red, svals, p_coefs, V_red = Exp.invert_measurement(r, strain, disps)
    
    Exp.contour_b_vec(inverted_b, 1, fringe=False, keep_boundaries=1, cbar=cbar, orientation='horizontal', title='reconstruction')
        
    
#%%

#
#
#
##
#
##
##

#%% make the experiment object and set meaurement scheme and model

for mod_name in ['20', '10','6.7','5']: # ,'6.7','5'
    
    nx = nxs[mod_name]
    ny = nys[mod_name]
    n = nx*ny
    Exp = Experiment_ortho2d_extra(data_obj, nx, ny, method='linear')
    
    # set measurement and model - E_y only and d_y only 
    Exp.set_special_measurement('all_y')
    mat = np.zeros((n*4, n))
    eye10 = np.eye(n)
    mat[n:2*n] = eye10
    Exp.set_model_matrix(Exp.double_sides_metric().dot(mat))
    
    m0 = Exp.compute_svd_at_b0(Exp.default_b_vector, strain)
    
    r = ranks[mod_name]
    
    inverted_b, d_coefs, U_red, svals, p_coefs, V_red = Exp.invert_measurement(r, strain, disps)
    # plot error field
    sigma = 0.003
    
    pvecs = Exp.param_s_vecs.copy()
    svals_inverse = np.reciprocal(np.asarray(Exp.s_vals))
    
    for ind in range(pvecs.shape[1]):
        pvecs[:, ind] *= svals_inverse[ind]*sigma
    
    squared = np.square(pvecs)    

    cutoff = cutoffs[mod_name]
    
    squared = squared[:, :cutoff]
    summed = np.sum(squared, axis=1)
    
    rooted = np.sqrt(summed)
    Exp.contour_b_vec(Exp.params_to_b(rooted), 1, keep_boundaries=0, cmap='Oranges', orientation='vertical')