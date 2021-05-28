import numpy as np

from sfepy.base.conf import ProblemConf
from sfepy.discrete.problem import Problem
from sfepy.mesh.mesh_generators import gen_block_mesh

from sfepy.postprocess.viewer import Viewer
from scipy.interpolate import interpn

import matplotlib.pyplot as plt



    
    

    



class Block2d:
    
    """ 
    Small wrapper for sfepy problem class.
    
    Constructs a rectangle of specificed dimensions with a rectangular FEM mesh.
    Rectangle is gripped at one end and pulled away at the other end, by a specified 
    distance. FEM mesh sixe can be specified. Centre of rectangle can be specified. 
    
    Having done this, this class exposes the stretch amount and the distribution 
    of the stiffness tensor D as paramters that can be set, and allows querying 
    of the displacement at a set of points in the domain. 
    
    rectangle is stretched parallel to the y-axis. short side of rectangle is x-axis. 
    
    Also drawing. """
    
    def __init__(self, dims, center_location, cell_sizes=np.array([2,2]),
                 prob_file="C:\\Users\\wbald\\sfepythings\\blocks\\fem\\prob_desc_2d.py", 
                 put_mesh="C:\\Users\\wbald\\sfepythings"):
        
        """ 
        dims:               array, dimensions of rectangle [x,y] 
        center_location:    array, centre of rectangle [x,y]
        cell sizes:         array, x and y side length of FEM rectangular elements
        stretch:            default distance by which to displace the upper y axis edge. 
        prob_file:          problem description file
        put_mesh:           where to save the mesh file
        """
        
        assert(dims.shape[0] == 2)
        assert(cell_sizes.shape[0] == 2)
        
        # assume linear elasticity. Fix strain of rectangle to 0.001 and query
        # at different strains by scaling linearly
        self.dims = dims
        self.prob_file = prob_file        
        self.FEM_model_strain = 0.001
        self.elongation= self.FEM_model_strain * self.dims[1]
        
        

        nums = np.divide(dims, cell_sizes)
        nums = np.around(nums).astype(int) + np.array([1,1])
        
        blockmesh = gen_block_mesh(dims, nums, center_location)
        blockmesh.write(put_mesh + '\\mesh.vtk')
        
        conf = ProblemConf.from_file(prob_file)
        
        # 'region_ylower__1' holds the edge to be fixed
        # 'region_yupper__2' holds the edge to be displaced
        conf.regions['region_ylower__1'].select = 'vertices in (y < ' + str(0.01) + ')'
        conf.regions['region_yupper__2'].select = 'vertices in (y > ' + str(dims[1] - 0.01) + ')'
        
        conf.ebcs['ebc_Displaced__1'].dofs['u.1'] = self.elongation
        
        self.prob = Problem.from_conf(conf)
        # for reshaping sfepy output into xyz displacements
        self.reshape_tuple = ((self.prob.fields['displacement'].n_nod, self.prob.fields['displacement'].n_components))        
        
    def _set_param_field(self, query):
        # the stiffness is specified by the voigt notation tensor D, which is 
        # passed to the problem class as a queryable function of position 
        def fun(ts, coors, mode=None, **kwargs):
            if mode=='qp':
                return {'D' : query(coors)}        
        self.prob.functions._objs[0].function = fun          
    
    def get_disp_at_coords(self, D_query_function, strain, coords):
        # get the displacement given a stiffness field D and displacement strain. 
        self._set_param_field(D_query_function)
        state = self.prob.solve()
        displacement = self.prob.fields['displacement'].evaluate_at(coords, state.vec.reshape(self.reshape_tuple))
        displacement *= strain / self.FEM_model_strain
        return displacement
    
    def drawstate(self, state):
        # draw the object with sfepy's viewer
        # state can be obtained from prob.solve()
        self.prob.save_state('curr_run_demo.vtk', state)
        view = Viewer('curr_run_demo.vtk')
        view(vector_mode='warp_norm', rel_scaling=2, is_scalar_bar=True, is_wireframe=True)  
     

    
    
    
class Experiment_ortho2d_extra:
    
    """ 
    Class for grouping a FEM model and a stiffness distribution model together, 
    performing reconstructions, visualising measurements and stiffness.
    
    This class deals with 4 material properties, ['E_x', 'E_y', '\nu_{xy}', 'G_{xy}'], 
    independently defined on a grid. 
    
    The class has a measurent scheme. This is specified by the dataobj object it holds, 
    which holds the details of a particular experiment (dimensions, strains, 
    location of measurement points, displacement measurements, stresses).
    
    A linear model can be specified which reduces the number of free paramters. 
    The class allows the displacement at any point to be queried
    """
    
    def __init__(self, dataobj, nx, ny, method='linear'):   
        # block and calcs
        # fringe is used to describe the ficticious material at each end of the coupon
        # block refers to the Block2d object holding the FEM model of the strip
        # strip_dimensions is the dimensions of the strip
        # the strip is placed centered on the y-axis, with the fixed coincident with the 
        # x-axis. 
        self.fringe = dataobj.fringe
        self.strip_dimensions = dataobj.object_estimated_dimensions + np.array([0, self.fringe*2])
        self.block = Block2d(self.strip_dimensions, np.array([0, self.strip_dimensions[1]/2]))
        
        # all_coordinates is the coordinates of all the the measured locations.
        # 2d
        self.all_coordinates = dataobj.centered_trunc_coords.copy()
        self.dshape = self.all_coordinates.shape
        self.n_points = self.all_coordinates.shape[0]
        
        # a measurement is a subset of the full list of measured coordinates
        self._measurement_scheme = np.array(range(self.n_points*2))
        
        # D tensor distribution. 
        # generated by bilinearly interpolating from a (nx by ny) grid. 
        # tensor is specfied by a 1d array called b_vector which is 4*nx*ny elements large
        self.nx = nx
        self.ny = ny
        self.x_vals = []
        self.y_vals = []
        self.interp_method = method
        self.D_interpolator_space, self.default_b_vector = self._setup_D_field_interpolation()        
        
        # model
        self.model_matrix = np.eye(4*self.nx*self.ny)
        self.default_p_vector = np.ones(4*self.nx*self.ny)
        self.coef_names = ['E_x', 'E_y', '\nu_{xy}', 'G_{xy}']
        
        # inversion
        self.inverted_at = []
        self.derivative = []
        self.param_s_vecs = []
        self.measurement_s_vecs = []
        self.s_vals = []
        
        
    """ ========================= D field function, models ======================= """
    
    def cs_from_ps(self, ps):
        # calc the D tensor elements from the four parameters. 
        cs = np.zeros(ps.shape)
        f_top = ps[0]
        f_bottom = ps[0] - np.multiply(ps[1],np.square(ps[2]))
        f = np.divide(f_top, f_bottom)
        
        cs[0] = np.multiply(ps[0] ,f)
        cs[1] = np.multiply(ps[1] ,f)
        cs[2] = ps[3]
        cs[3] = np.multiply(np.multiply(ps[1], f), ps[2])    
        return cs   

    def _setup_D_field_interpolation(self):        
        self.x_vals = np.linspace(-self.strip_dimensions[0]/2, self.strip_dimensions[0]/2, self.nx)
        self.y_vals = np.linspace(0, self.strip_dimensions[1], self.ny)
        
        def func(query_coords, b_vec):
            # will reshape b_vec into params which is d*nx*ny
            # convention is to use the 'C' ordering
            # this means we will count in y before in x.
            
            # parameters: p1 ... p4.
            
            params = b_vec.reshape((4, self.nx, self.ny))
            
            oneoneT = np.array([[1,0,0],[0,0,0],[0,0,0]])
            twotwoT = np.array([[0,0,0],[0,1,0],[0,0,0]])
            thrthrT = np.array([[0,0,0],[0,0,0],[0,0,1]])
            symparT = np.array([[0,1,0],[1,0,0],[0,0,0]])
            
            x_vals = self.x_vals
            y_vals = self.y_vals
            
            cs = self.cs_from_ps(params)
            
            C11 = interpn((x_vals, y_vals), cs[0, :, :], query_coords, method=self.interp_method)
            C22 = interpn((x_vals, y_vals), cs[1, :, :], query_coords, method=self.interp_method)
            C33 = interpn((x_vals, y_vals), cs[2, :, :], query_coords, method=self.interp_method)
            C12 = interpn((x_vals, y_vals), cs[3, :, :], query_coords, method=self.interp_method)
            
            C11enlarge = np.array(C11)[...,None,None]
            C22enlarge = np.array(C22)[...,None,None]
            C33enlarge = np.array(C33)[...,None,None]
            C12enlarge = np.array(C12)[...,None,None]
            
            Mat = C11enlarge*oneoneT + C22enlarge*twotwoT + C33enlarge*thrthrT + C12enlarge*symparT            
            return Mat
        
        # get default params      
        psdef = np.array([1, 1, 0.3, 0.38])
        ones_shape = np.ones((self.nx, self.ny))
        c11def = psdef[0]*ones_shape
        c22def = psdef[1]*ones_shape
        c33def = psdef[2]*ones_shape
        c12def = psdef[3]*ones_shape
                
        default = np.array([c11def, c22def, c33def, c12def])
        default = default.reshape(4*self.nx*self.ny)
        
        return func, default
    
    def make_D_interpolator(self, b_vector):
        return lambda coords : self.D_interpolator_space(coords, b_vector)
    
    def params_to_b(self, params):
        return self.model_matrix.dot(params)
    
    def set_model_matrix(self, matrix):
        self.model_matrix = matrix
        self.default_p_vector = np.ones(matrix.shape[1])
        
    """ ========================= displacements and measeurements ========================= """
    
    def get_displacement(self, strain, b_vector):
        D_interpolator = self.make_D_interpolator(b_vector)
        displacement = self.block.get_disp_at_coords(D_interpolator, strain, self.all_coordinates)
        return displacement
    
    def get_measurement(self, strain, b_vec):
        disp = self.get_displacement(strain, b_vec)
        return self.measurement_from_displacement(disp)
        
    def set_measurement_scheme(self, indices):
        self._measurement_scheme = indices        
        
    def set_special_measurement(self, measurement_type):
        accepted = ['all_x', 'all_y']
        assert measurement_type in accepted        
        if measurement_type == 'all_x':
            indices = np.arange(0, 2*self.n_points, 2)
        else:
            indices = np.arange(1, 2*self.n_points, 2)            
        self.set_measurement_scheme(indices)  
    
    def measurement_from_displacement(self, displacement):
        # extracts from a displacement array the data correpsonding to the 
        # measured degrees of freedom as specified by the current measurement scheme.
        flat_displacement = displacement.flatten()
        return flat_displacement[self._measurement_scheme]

    def embed_measurement_in_displacement(self, measurement):
        assert measurement.shape == self._measurement_scheme.shape
        displacement = np.zeros(self.n_points*2)
        displacement[self._measurement_scheme] = measurement
        displacement = displacement.reshape(self.dshape)
        return displacement
        
    """ ========================= inverting ========================= """
    
    def compute_derivative(self, b_vec0, strain):
        # computes derivative w.r.t the model and measurement scheme.
        measurement0 = self.get_measurement(strain, b_vec0)
        num_params = self.model_matrix.shape[1]
        derivative = np.zeros((self._measurement_scheme.shape[0], num_params))
        delta = 0.001
        for index in range(num_params):
            p_perturbed = np.zeros(num_params)
            p_perturbed[index] = delta
            b_perturbed = b_vec0.copy() + self.model_matrix.dot(p_perturbed)
            m_perturbation = self.get_measurement(strain, b_perturbed) - measurement0
            derivative[:, index] = m_perturbation / delta
            
        return derivative, measurement0
    
    def restricted_derivative_wrt_svecs(self, b, num_svecs, strain):
        # computes derivative w.r.t the model and measurement scheme
        # at the point b and only in the place spanned by the first n b-svecs.
        # each col corresponds to a s-vec
        measurement0 = self.get_measurement(strain, b)
        derivative = np.zeros((self._measurement_scheme.shape[0], num_svecs))
        delta = 0.01
        for index in range(num_svecs):
            p_perturbed = delta*self.param_s_vecs[:, index]
            b_perturbed = b.copy() + self.model_matrix.dot(p_perturbed)
            m_perturbation = self.get_measurement(strain, b_perturbed) - measurement0
            derivative[:, index] = m_perturbation / delta
            
        return derivative
    
    def compute_svd_at_b0(self, b_vec, strain):
        self.inverted_at = b_vec
        der_mat, measurement0 = self.compute_derivative(b_vec, strain)
        self.derivative = der_mat.copy()
        
        U, s, Vt = np.linalg.svd(der_mat)
        self.measurement_s_vecs = U
        self.s_vals = s
        self.param_s_vecs = Vt.transpose()
        
        return measurement0
        
    def double_sides_metric(self):
        Q = np.eye(self.default_b_vector.shape[0])
        
        # all params on a wall get doubled. ignore corners...
        for index in range(self.default_b_vector.shape[0]):
            xy_index = index % (self.nx*self.ny)
            # yindex = xy_index % self.nx
            xindex = xy_index // self.ny
            if xindex == 0 or xindex == self.nx - 1:
                Q[index, index] = 2.0**0.5
        return Q
        
    def invert_measurement(self, rank, strain, measurement):
        """ performs the linear FEMU inversion """
        
        # reduced rank matrices
        U_red = self.measurement_s_vecs[:, :rank]
        s_inv = np.reciprocal(self.s_vals)
        S_inv_red = np.diag(s_inv[:rank])
        V_red = self.param_s_vecs[:, :rank]
        
        # d side perturbation
        measurement_perturbation = measurement - self.get_measurement(strain, self.inverted_at)
        
        # reduced rank expansion coefs
        d_side_coefs = U_red.transpose().dot(measurement_perturbation)
        p_side_coefs = S_inv_red.dot(d_side_coefs)     
        
        # param perturbation
        params_perturbation = V_red.dot(p_side_coefs)
        inferred_b_vec = self.inverted_at + self.model_matrix.dot(params_perturbation)
        
        return inferred_b_vec, d_side_coefs, U_red, self.s_vals[:rank], p_side_coefs, V_red
        
    def iterate_stiffness(self, rank, strain, measurement, num_iterations, step):
        # freezes the singular vectors. 
        # steps towards solution.
        current_b = self.default_b_vector
        bs = []
        ms = []
        for itr in range(num_iterations):
            current_m = self.get_measurement(strain, current_b)
            ms.append(current_m)
            bs.append(current_b)
            
            delta_m = measurement - current_m
            A = self.restricted_derivative_wrt_svecs(current_b, rank, strain)
            trunc_vecs_coefs_prime, _, _, _ = np.linalg.lstsq(A, delta_m)
            
            V = self.param_s_vecs
            delta_b = np.zeros(self.default_b_vector.shape[0])
            for ind in range(trunc_vecs_coefs_prime.shape[0]):
                delta_b += self.params_to_b( V[:, ind] * trunc_vecs_coefs_prime[ind] )
            
            current_b = current_b + delta_b*step        
                        
        return bs, ms               
        
    def invert_tikhonov(self, rank, strain, measurement, alpha, return_t='b'):
        assert return_t in ['b', 'p', 'all']
        
        # reduce the rank
        U_red = self.measurement_s_vecs[:, :rank]
        s_inv = np.reciprocal(self.s_vals)
        S_inv_red = np.diag(s_inv[:rank])
        V_red = self.param_s_vecs[:, :rank]
        
        measurement_perturbation = measurement - self.get_measurement(strain, self.inverted_at)
        phis = np.zeros(rank)
        for i in range(rank):
            phis[i] = self.s_vals[i]**2 / (self.s_vals[i]**2 + alpha**2)
        
        d_side_coefs = U_red.transpose().dot(measurement_perturbation)
        p_side_coefs = S_inv_red.dot(d_side_coefs)        
        p_side_coefs = np.multiply(p_side_coefs, phis)
        params_perturbation = V_red.dot(p_side_coefs)
        
        if return_t=='b':
            inferred_b_vec = self.inverted_at + self.model_matrix.dot(params_perturbation)
            return inferred_b_vec     
        elif return_t=='p':
            return params_perturbation
        else:
            list_of_svecs = [vec for vec in V_red.transpose()]
            inferred_b_vec = self.inverted_at + self.model_matrix.dot(params_perturbation)
            return inferred_b_vec, params_perturbation, list_of_svecs, self.s_vals, p_side_coefs, d_side_coefs
        
    """ ========================= plotting ========================= """
    
    def _calc_xy_measured(self, dof_measurement_indices):
        # need to check through to see if every node has an x or a y disp active, for plotting.
        measured_x_points = []
        measured_y_points = []
        
        for index in dof_measurement_indices:
            if index % 2 == 0:
                measured_x_points.append(index//2)
            if index % 2 == 1:
                measured_y_points.append(index//2)
                
        return np.array(measured_x_points), np.array(measured_y_points)
        
    def scatter_displacement(self, displacement, direction):
        plt.figure()
        plt.scatter(self.all_coordinates[:,1], self.all_coordinates[:,0], c=displacement[:,direction])
        plt.xlim([0, self.strip_dimensions[1]])
        plt.ylim([-self.strip_dimensions[0]/2, self.strip_dimensions[0]/2])
        plt.colorbar()
        plt.show()
        
    def scatter_measurement(self, measurement, direction, fringe=True, cbar=None, orientation='horizontal', **kwargs):
        assert direction in [0,1]
        assert measurement.shape == self._measurement_scheme.shape
        
        all_displacements = self.embed_measurement_in_displacement(measurement)         
        measured_x_points, measured_y_points = self._calc_xy_measured(self._measurement_scheme)
        
        if direction == 0:
            assert measured_x_points.shape[0] > 0            
            coordinates = self.all_coordinates[measured_x_points, :]
            used_displacements = all_displacements[measured_x_points, 0]
            
        if direction == 1:
            assert measured_y_points.shape[0] > 0            
            coordinates = self.all_coordinates[measured_y_points, :]
            used_displacements = all_displacements[measured_y_points, 1]
        
        #plt.figure()        
        #plt.colorbar()
        #plt.show()        
        
        fig, ax = plt.subplots(figsize=(15,4))
        
        thing = ax.scatter(coordinates[:,1], coordinates[:,0], c=used_displacements, s=5, cmap='plasma', **kwargs)
        plt.xlim([0, self.strip_dimensions[1]])
        plt.ylim([-self.strip_dimensions[0]/2, self.strip_dimensions[0]/2])
        if not fringe:
            plt.xlim([self.fringe, self.strip_dimensions[1]-self.fringe])
        
        ax.set_xlabel('y (mm)')
        ax.set_ylabel('x (mm)')
        
        if cbar is None:            
            fig.colorbar(thing, orientation=orientation)
            ax.set_aspect('equal')
        else:
            fig.colorbar(thing, orientation=orientation)
            thing.set_clim(cbar.vmin, cbar.vmax)
            ax.set_aspect('equal')
            
        if orientation=='vertical':
            thing.colorbar.ax.set_ylabel('y-displacement (mm)', labelpad=15)
        else:
            thing.colorbar.ax.set_xlabel('y-displacement (mm)', labelpad=15)
      
        plt.show() 
        return thing.colorbar
        
        
        
    def scatter_measurement_1d(self, measurement1, measurement2, direction):
        assert direction in [0,1]
        assert measurement1.shape == self._measurement_scheme.shape
        measured_x_points, measured_y_points = self._calc_xy_measured(self._measurement_scheme)
        
        all_displacements = self.embed_measurement_in_displacement(measurement1)        
        
        if direction == 0:
            assert measured_x_points.shape[0] > 0            
            coordinates = self.all_coordinates[measured_x_points, :]
            used_displacements = all_displacements[measured_x_points, 0]
            
        if direction == 1:
            assert measured_y_points.shape[0] > 0            
            coordinates = self.all_coordinates[measured_y_points, :]
            used_displacements = all_displacements[measured_y_points, 1]
        
        plt.figure()
        plt.scatter(coordinates[:,1], used_displacements, s=0.1)
        
        assert measurement2.shape == self._measurement_scheme.shape        
        all_displacements = self.embed_measurement_in_displacement(measurement2)
        
        if direction == 0:
            assert measured_x_points.shape[0] > 0            
            coordinates = self.all_coordinates[measured_x_points, :]
            used_displacements = all_displacements[measured_x_points, 0]
            
        if direction == 1:
            assert measured_y_points.shape[0] > 0            
            coordinates = self.all_coordinates[measured_y_points, :]
            used_displacements = all_displacements[measured_y_points, 1]
        
        plt.scatter(coordinates[:,1], used_displacements, s=0.1)
        
        plt.xlim([0, self.strip_dimensions[1]])
        plt.colorbar()
        plt.show()
        
    def plot_b_vec(self, b_vec, coef_index, fringe=False):
        field = b_vec.reshape((4, self.nx, self.ny))[coef_index]
        
        plot_density = 1.        
        if fringe:
            plot_ranges = [-self.strip_dimensions[0]/2, self.strip_dimensions[0]/2, 0, self.strip_dimensions[1]]
        else:
            plot_ranges = [-self.strip_dimensions[0]/2, self.strip_dimensions[0]/2, self.fringe, self.strip_dimensions[1] - self.fringe]        
        xr = np.arange(plot_ranges[0], plot_ranges[1], plot_density)
        yr = np.arange(plot_ranges[2], plot_ranges[3], plot_density)
        X, Y = np.meshgrid(xr, yr)
        vals = interpn((self.x_vals, self.y_vals), field, np.array([X.flatten(), Y.flatten()]).transpose())
        
        fig, ax = plt.subplots()
        thing = ax.contourf(Y,X,vals.reshape(X.shape), 100)
        fig.colorbar(thing)
        ax.set_aspect('equal')
        plt.show()  
        
    def contour_b_vec(self, b_vec, coef_index, cbar=None, fringe=False, title='', keep_boundaries=0, orientation='horizontal', **kwargs):
        field = b_vec.reshape((4, self.nx, self.ny))[coef_index]
        
        plot_density = 1.        
        if fringe:
            plot_ranges = [-self.strip_dimensions[0]/2, self.strip_dimensions[0]/2, 0, self.strip_dimensions[1]]
        else:
            plot_ranges = [-self.strip_dimensions[0]/2, self.strip_dimensions[0]/2, self.fringe, self.strip_dimensions[1] - self.fringe]   
        xr = np.arange(plot_ranges[0], plot_ranges[1], plot_density)
        yr = np.arange(plot_ranges[2], plot_ranges[3], plot_density)
        X, Y = np.meshgrid(xr, yr)
        vals = interpn((self.x_vals, self.y_vals), field, np.array([X.flatten(), Y.flatten()]).transpose())
        
        fig, ax = plt.subplots(figsize=(15,4))        
        ax.set_aspect('equal')
        
        levels = 15
        
        if (cbar is not None) and (keep_boundaries):    
            levels = cbar._boundaries
            
        thing = ax.contourf(Y,X,vals.reshape(X.shape), **kwargs, levels=levels)  
        fig.colorbar(thing, orientation=orientation)
        
        if (cbar is not None) and (not keep_boundaries):   
            thing.set_clim(cbar.vmin, cbar.vmax)
        
        ax.set_xlabel('y (mm)')
        ax.set_ylabel('x (mm)')
        # thing.colorbar.ax.set_ylabel(self.coef_names[coef_index])       
        plt.title(title)
        
        plt.show() 
        
        return thing.colorbar
        
    def plot_p_vec(self, p_vec, coef_index, **kwargs):
        b_vec = self.model_matrix.dot(p_vec)
        self.plot_b_vec(b_vec, coef_index, **kwargs)
        
        
        
        