import numpy as np
from scipy.stats import linregress

from blocks.datatools.fileReading import (EXP_NAMES, FOLDER_PATHS, BLOCK_DIMS, BLOCK_AXES_OFFSETS, 
                                    unioned_folder_displacement, get_folder_stress)



def regress_to_stress(disps, stress):
    
    """ Takes a dictionary of disp-shaped arrays and a dict of measured stresses.
        linearly regeresses each dispalacement degree of freedom to the stress value.
        returns a gradient and offset and variance vector """
    
    assert type(stress) == dict
    assert type(disps) == dict
    
    keys = stress.keys()
    stresses = []
    all_disps = []
    for key in keys:
        stresses.append(stress[key])
        long_vec = disps[key].flatten(order='F')
        all_disps.append(long_vec)
    
    all_disps = np.array(all_disps)
    num_dofs = all_disps.shape[1]
    
    slopes = np.zeros(num_dofs)
    intercepts = np.zeros(num_dofs)
    variances = np.zeros(num_dofs)
    
    t_variances = np.var(all_disps, axis=0)
    
    for ind in range(num_dofs):    
        slopes[ind], intercepts[ind], r_val, p_val, stderr = linregress(stresses, all_disps[:, ind])
        variances[ind] = (1 - r_val)*t_variances[ind]

    slopes = slopes.reshape((num_dofs//2, 2), order='F')
    intercepts = intercepts.reshape((num_dofs//2, 2), order='F')
    variances = variances.reshape((num_dofs//2, 2), order='F')
        
    return slopes, intercepts, variances



def regress_to_strain(disps, strains):
    """ will return a gradient and offset verctor, as well as a std dev for the average mesasurment """
    assert type(strains) == dict
    assert type(disps) == dict
    
    keys = strains.keys()
    
    # get all strains, flattened displacements
    # all disps will be an array (#steps, #dofs)
    strain_list = []
    all_disps = []
    for key in keys:
        strain_list.append(strains[key])
        long_vec = disps[key].flatten(order='F')
        all_disps.append(long_vec)
        
        #
    strain_list = np.asarray(strain_list)
    
    all_disps = np.array(all_disps)
    num_dofs = all_disps.shape[1]
    
    slopes = np.zeros(num_dofs)
    intercepts = np.zeros(num_dofs)
    variances = np.zeros(num_dofs)
    
    t_variances = np.var(all_disps, axis=0)
    
    #for ind in range(num_dofs):    
    #    slopes[ind], intercepts[ind], r_val, p_val, stderr = linregress(strain_list, all_disps[:, ind])
    #    variances[ind] = (1 - r_val)*t_variances[ind]
    
    for ind in range(num_dofs):    
        slopes[ind], _, _, _ = np.linalg.lstsq(strain_list[:, np.newaxis], all_disps[:, ind])
        intercepts[ind] = 0
        variances[ind] = 0

    slopes = slopes.reshape((num_dofs//2, 2), order='F')
    intercepts = intercepts.reshape((num_dofs//2, 2), order='F')
    variances = variances.reshape((num_dofs//2, 2), order='F')
        
    return slopes, intercepts, variances





def regress_to_dict(disps, predictor, force_zero):
    
    """ Takes a dictionary of disp-shaped arrays and a dict of measured stresses.
        linearly regeresses each dispalacement degree of freedom to the stress value.
        returns a gradient and offset and variance vector """
    
    assert force_zero in [True, False]
    assert type(predictor) == dict
    assert type(disps) == dict
    keys_list = lambda dictionary : [key for key in dictionary.keys()]
    assert keys_list(disps) == keys_list(predictor)
    
    keys = keys_list(predictor)
    pred_arr = []
    disps_arr = []
    for key in keys:
        pred_arr.append(predictor[key])
        long_vec = disps[key].flatten(order='F')
        disps_arr.append(long_vec)
    
    disps_arr = np.asarray(disps_arr)
    num_dofs = disps_arr.shape[1]
    pred_arr = np.asarray(pred_arr)
    
    slopes = np.zeros(num_dofs)
    intercepts = np.zeros(num_dofs)
    variances = np.zeros(num_dofs)
    total_variances = np.var(disps_arr, axis=0)     
    
    if force_zero:
        for ind in range(num_dofs):    
            slopes[ind], res, _, _ = np.linalg.lstsq(pred_arr[:, np.newaxis], disps_arr[:, ind])
            intercepts[ind] = 0
            variances[ind] = res
        
    else:           
        for ind in range(num_dofs):    
            slopes[ind], intercepts[ind], r_val, p_val, stderr = linregress(pred_arr, disps_arr[:, ind])
            variances[ind] = (1 - r_val**2)*total_variances[ind]
    
    slopes = slopes.reshape((num_dofs//2, 2), order='F')
    intercepts = intercepts.reshape((num_dofs//2, 2), order='F')
    variances = variances.reshape((num_dofs//2, 2), order='F')
        
    return slopes, intercepts, variances




def gradient_vector_svd(disps):
    """ will return a gradient verctor """
    assert type(disps) == dict
    
    # we solve by finding the principle axis of the data matrix A = (d1, d2, ... )
    # only use y-displacement for now
    
    A = []
    for vec in disps.values():
        A.append(vec[:, 1])
    A = np.asarray(A).transpose()
    
    V, s, Ut = np.linalg.svd(A)
    grad_vec = V[:, 0]
    return grad_vec
    
def gradient_vector2d_svd(disps):
    """ will return a gradient verctor """
    assert type(disps) == dict
    
    # we solve by finding the principle axis of the data matrix A = (d1, d2, ... )
    
    
    A = []
    for vec in disps.values():
        A.append(vec.flatten())
    A = np.asarray(A).transpose()
    
    V, s, Ut = np.linalg.svd(A)
    grad_vec = V[:, 0]
    grad_vec = grad_vec.reshape(vec.shape)
    return grad_vec
    


def truncate_and_shift(coords, disps, include_ranges):    
    # doesnt actually shift
    assert type(disps) == dict
    
    x_lower = include_ranges[0]
    x_upper = include_ranges[1]
    y_lower = include_ranges[2]
    y_upper = include_ranges[3]
    
    num_points = coords.shape[0]
    
    trunc_coords = []
    trunc_disps = {}
    
    for key in disps.keys():
        trunc_disps[key] = []
    
    for index in range(num_points):
        if coords[index, 0] < x_upper and coords[index, 0] > x_lower and coords[index, 1] < y_upper and coords[index, 1] > y_lower:
            trunc_coords.append(coords[index, :])
            for key in disps.keys():
                trunc_disps[key].append(disps[key][index, :])
    
    trunc_coords = np.asarray(trunc_coords)
    for key in disps.keys():       
        trunc_disps[key] = np.asarray(trunc_disps[key])
        
    return trunc_coords, trunc_disps






def re_center_measurement(fringe, data_block_ranges, raw_coordinates, raw_displacement):
    assert type(raw_displacement) == np.ndarray
        
    """takes a set of displacements, and computes the appropriate linear trend to make a model
    also shifts the displacement values and the coordinates to match this experiment. 
    returns the new coordinates, displacements, and the average strain"""
        
    new_coordinates = raw_coordinates.copy()
    new_displacements = raw_displacement.copy()
    
    # centre coords
    # block lies between data_block_ranges[0] and data_block_ranges[1] in x, simlar for y
    new_coordinates[:, 0] -= (data_block_ranges[0] + data_block_ranges[1]) / 2
    new_coordinates[:, 1] -= data_block_ranges[2]

    
    # find a linear fit to the y-disp data in order to make the experiement setup
    strain, intercept, r_value, p_value, std_err = linregress(new_coordinates[:, 1], 
                                                             new_displacements[:, 1])
    
    # then we move the coordiates over by edge, and increase the y dispalcements to a linear fit with zero intercept
    new_coordinates[:, 1] += fringe      
    new_displacements[:, 1] -= intercept - fringe*strain
    
    print((intercept - fringe*strain)/strain)
    
    # finally remove linear and constant variation from x displacement
    #slopex, interceptx, r_value, p_value, std_err = linregress(new_coordinates[:, 1], new_displacements[:, 0])
    #linear_component = new_coordinates[:, 1]*slopex + interceptx
    #new_displacements[:, 0] -= linear_component        
    
    return new_coordinates, new_displacements, strain            
        
        
        
class ExpData2:
    def __init__(self, exp_name, fringe=10):        
        # check exists get dims
        assert exp_name in EXP_NAMES        
        self.name = exp_name
        
        self.object_estimated_dimensions = BLOCK_DIMS[self.name][:2]        
        dims = BLOCK_DIMS[self.name][:2]
        zeros = BLOCK_AXES_OFFSETS[self.name]        
        self.data_ranges = [zeros[0], zeros[0] + dims[0], zeros[1], zeros[1] + dims[1]]
        
        self.fringe = fringe
        
        # read matched data, inc stresses, fill frames with frame names
        self.all_coords, self.all_disps_dict = unioned_folder_displacement(FOLDER_PATHS[self.name])
        self.stresses_dict = get_folder_stress(FOLDER_PATHS[self.name])        
        self.frames = [key for key in self.stresses_dict.keys()]
        
        ## new bit!!
        print(self.frames)
        f = self.frames.pop(0)
        self.all_disps_dict.pop(f)
        self.stresses_dict.pop(f)
        print(self.frames)
        
        ## old bit!!
        
        # truncate the coordinates to data_ranges
        self.truncated_coordinates, self.truncated_displacements = truncate_and_shift(self.all_coords,
                                                                                 self.all_disps_dict,
                                                                                 self.data_ranges)
                
        # shift the disps and coords so that each one is intercept at zero, save the average strains.
        self.centered_trunc_disps = {}
        self.strains = {}      
        for frame in self.frames:
            c,d,s = re_center_measurement(self.fringe, 
                                         self.data_ranges, 
                                         self.truncated_coordinates, 
                                         self.truncated_displacements[frame])
            self.centered_trunc_coords = c
            self.centered_trunc_disps[frame] = d
            self.strains[frame] = s      
            
        # regress the new displacements against their strains
        self.gradient_vector, self.intercept_vector, self.variances = regress_to_strain(self.centered_trunc_disps, self.strains)
        
        # calculate the noise component that woudl be in each vector
        # if each vector is the mean plus noise
        self.residuals = {}
        for frame in self.frames[:-1]:
            residuals = self.centered_trunc_disps[frame] - self.gradient_vector * self.strains[frame]
            self.residuals[frame] = residuals




class resultsSet:
    def __init__(self, exp_name, fringe=10):  
        # check exists get dims
        assert exp_name in EXP_NAMES        
        self.name = exp_name
        
        self.object_estimated_dimensions = BLOCK_DIMS[self.name][:2]        
        dims = BLOCK_DIMS[self.name][:2]
        zeros = BLOCK_AXES_OFFSETS[self.name]        
        self.data_ranges = [zeros[0], zeros[0] + dims[0], zeros[1], zeros[1] + dims[1]]
        
        self.fringe = fringe
        
        # read matched data, inc stresses, fill frames with frame names
        self.all_coords, self.all_disps_dict = unioned_folder_displacement(FOLDER_PATHS[self.name])
        self.stresses_dict = get_folder_stress(FOLDER_PATHS[self.name])        
        self.frames = [key for key in self.stresses_dict.keys()]
        
        # truncate the coordinates to data_ranges
        self.truncated_coordinates, self.truncated_displacements = truncate_and_shift(self.all_coords,
                                                                                 self.all_disps_dict,
                                                                                 self.data_ranges)
                
        # shift the disps and coords so that each one is intercept at zero, save the average strains.
        self.centered_trunc_disps = {}
        self.strains = {}        
        for frame in self.frames:
            c,d,s = re_center_measurement(self.fringe, 
                                         self.data_ranges, 
                                         self.truncated_coordinates, 
                                         self.truncated_displacements[frame])
            self.centered_trunc_coords = c
            self.centered_trunc_disps[frame] = d
            self.strains[frame] = s      
            
        # regress the new displacements against their strains
        self.gradient_vector, self.intercept_vector, self.variances = regress_to_strain(self.centered_trunc_disps, self.strains)
        
        # calculate the noise component that woudl be in each vector
        # if each vector is the mean plus noise
        self.residuals = {}
        for frame in self.frames[:-1]:
            residuals = self.centered_trunc_disps[frame] - self.gradient_vector * self.strains[frame]
            self.residuals[frame] = residuals
            
            
