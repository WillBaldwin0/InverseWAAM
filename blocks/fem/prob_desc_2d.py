from __future__ import absolute_import
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson

me = 'C:\\Users\\wbald\\sfepythings\\'
filename_mesh = me + 'mesh.vtk'


# default function is constant. 
def get_D(ts, coors, mode=None, **kwargs):
    if mode=='qp':
        val = stiffness_from_youngpoisson(2, 1, 0.27)
        #val = stiffness_from_lame(dim=3, 
        #                          lam=1e1*np.ones(coors.shape[0]), 
        #                          mu=1e0*np.ones(coors.shape[0]))
        return {'D' : val}

    
regions = {
    'Omega' : 'all',
    'ylower' : ('vertices in (y < -7.49)', 'facet'),
    'yupper' : ('vertices in (y > 7.49)', 'facet'),
}

materials = {
    'mat' : 'get_D',
}

functions = {
    'get_D' : (get_D,)
}

fields = {
    'displacement': ('real', 'vector', 'Omega', 1),
}

integrals = {
    'i' : 1,
}

variables = {
    'u' : ('unknown field', 'displacement', 0),
    'v' : ('test field', 'displacement', 'u'),
}

ebcs = {
    'Fixed' : ('ylower', {'u.all' : 0.0}),
    'Displaced' : ('yupper', {'u.1' : 0.1, 'u.0' : 0.0}),
}

equations = {
    'balance_of_forces' :
    'dw_lin_elastic.i.Omega(mat.D, v, u) = 0',
}

solvers = {
    'ls': ('ls.auto_direct', {}),
    'newton': ('nls.newton', {
        'i_max'      : 1,
        'eps_a'      : 1e-10,
    }),
}
