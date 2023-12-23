import numpy as np
import torch as th

def get_decay_sin(x, tau, w):
    if type(x) is np.ndarray:
        base = np.exp(-x/tau) * np.sin(w * x)
    else:
        base = th.exp(-x/tau) * th.sin(w * x)
    # norm = (base**2).sum()**0.5
    # base = base / norm
    return base

def get_decay_cos(x, tau, w):
    if type(x) is np.ndarray:
        base = np.exp(-x/tau) * np.cos(w * x)
    else:
        base = th.exp(-x/tau) * th.cos(w * x)
    # norm = (base**2).sum()**0.5
    # base = base / norm
    return base

def get_decay_cos_cumsum(x, tau, w):
    sin = th.sin if th.is_tensor(x) else np.sin
    cos = th.cos if th.is_tensor(x) else np.cos
    exp = th.exp if th.is_tensor(x) else np.exp
    
    result = exp(-x/tau) * (w * sin(w *x) - 1/tau * cos(w*x)) + 1/tau
    result /= (1/tau**2 + w**2)
    return result

def get_decay_sin_cumsum(x, tau, w):
    sin = th.sin if th.is_tensor(x) else np.sin
    cos = th.cos if th.is_tensor(x) else np.cos
    exp = th.exp if th.is_tensor(x) else np.exp
    
    result = exp(-x/tau) * (1/tau * sin(w *x) + w * cos(w*x)) - w
    result /= - (1/tau**2 + w**2)
    return result

def get_decay_basis(x,tau, ws ):
    # basis_cos = np.exp(-x/tau)[None,:] * np.cos(ws[:,None] * x[None,:])
    # basis_sin = np.exp(-x/tau)[None,:] * np.sin(ws[1:,None] * x[None,:])
    # return np.concatenate([basis_cos, basis_sin], 0)
    basis = [get_decay_cos(x,tau,w) for w in ws] + [get_decay_sin(x,tau,w) for w in ws ]
    # basis = [get_decay_cos(x,tau,w) for w in ws] + [get_decay_sin(x,tau,w) for w in ws]
    nbasis = len(basis)
    if type(x) is np.ndarray:
        return np.array(basis)
    else:
        return th.cat(basis, 0).reshape(nbasis, -1)

     
def int_exp_cos_cos(a,b,c,x):
    """
    indefinite integral of exp(-ax)cos(bx)cos(cx)
    """
    sin = th.sin if th.is_tensor(x) else np.sin
    cos = th.cos if th.is_tensor(x) else np.cos
    exp = th.exp if th.is_tensor(x) else np.exp
    
    result  = ((b-c) * sin((b-c)*x) - a * cos((b-c)*x)) / (a**2+(b-c)**2)
    result += ((b+c) * sin((b+c)*x) - a * cos((b+c)*x)) / (a**2+(b+c)**2)
    result *= 0.5 * exp(-a*x)
    return result

def period_int_exp_cos_cos(a,b,c):
    """
    integral of exp(-ax)cos(bx)cos(cx) from 0 to z (z->infinity, bz, cz is multiple of 2pi) 
    i.e.
    int_exp_cos_cos(a, b, c, z) - int_exp_cos_cos(a, b, c, 0)
    """    
    result  = 1 / (a**2+(b-c)**2)
    result += 1 / (a**2+(b+c)**2)
    result *=  0.5 * a 
    return result

def int_exp_cos_sin(a,b,c,x):
    """
    indefinite integral of exp(-ax)cos(bx)cos(cx)
    """
    sin = th.sin if th.is_tensor(x) else np.sin
    cos = th.cos if th.is_tensor(x) else np.cos
    exp = th.exp if th.is_tensor(x) else np.exp
    
    result  = (a * sin((b-c)*x) + (b-c) * cos((b-c)*x)) / (a**2+(b-c)**2)
    result -= (a * sin((b+c)*x) + (b+c) * cos((b+c)*x)) / (a**2+(b+c)**2)
    result *= 0.5 * exp(-a*x)
    return result

def period_int_exp_cos_sin(a,b,c):
    """
    integral of exp(-ax)cos(bx)sin(cx) from 0 to z (z->infinity, bz, cz is multiple of 2pi) 
    i.e.
    int_exp_cos_sin(a, b, c, z) - int_exp_cos_sin(a, b, c, 0)
    """    
    result  = - (b-c) / (a**2+(b-c)**2)
    result += + (b+c) / (a**2+(b+c)**2)
    result *=  0.5
    return result

def int_exp_sin_cos(a,b,c,x):
    return int_exp_cos_sin(a,c,b,x)

def period_int_exp_sin_cos(a,b,c):
    """
    integral of exp(-ax)sin(bx)cos(cx) from 0 to z (z->infinity, bz, cz is multiple of 2pi) 
    """    
    return period_int_exp_cos_sin(a,c,b)


def int_exp_sin_sin(a,b,c,x):
    """
    indefinite integral of exp(-ax)cos(bx)cos(cx)
    """
    sin = th.sin if th.is_tensor(x) else np.sin
    cos = th.cos if th.is_tensor(x) else np.cos
    exp = th.exp if th.is_tensor(x) else np.exp
    
    result  = ((b-c) * sin((b-c)*x) - a * cos((b-c)*x)) / (a**2+(b-c)**2)
    result += (a * cos((b+c)*x) - (b+c) * sin((b+c)*x)) / (a**2+(b+c)**2)
    result *= 0.5 * exp(-a*x)
    return result

def period_int_exp_sin_sin(a,b,c):
    """
    integral of exp(-ax)sin(bx)sin(cx) from 0 to z (z->infinity, bz, cz is multiple of 2pi) 
    i.e.
    int_exp_sin_sin(a, b, c, z) - int_exp_sin_sin(a, b, c, 0)
    """    
    result  = 1 / (a**2+(b-c)**2)
    result -= 1 / (a**2+(b+c)**2)
    result *=  0.5 * a 
    return result



####################   not TORCH compatible, archived #######################3
    
# def decay_cos_kernel(tau, omega, cutoff, dt ):
#     mesh_size = int(tau/dt * cutoff)
#     t_mesh = (np.arange(mesh_size)+0.5) * dt
#     kernel = np.exp(-t_mesh/tau) * np.cos(omega * t_mesh)
#     return kernel

# def decay_sin_kernel(tau, omega, cutoff, dt ):
#     mesh_size = int(tau/dt * cutoff)
#     t_mesh = (np.arange(mesh_size)+0.5) * dt
#     kernel = np.exp(-t_mesh/tau) * np.sin(omega * t_mesh)
#     return kernel

# def get_xi_process(white_noise, tau, omega, dt, kernel_cutoff=5):
#     ckernel = decay_cos_kernel(tau, omega, kernel_cutoff, dt)
#     skernel = decay_sin_kernel(tau, omega, kernel_cutoff, dt)
#     xi_cos = np.convolve(white_noise, ckernel, mode='valid')
#     xi_sin = np.convolve(white_noise, skernel, mode='valid')
#     return xi_cos, xi_sin

# def get_xi_process_spectrum(tau, ws, dt,  length, cutoff=5):
#     '''
#     Args:
#         tau: the lifetime of the xi process
#         ws: the frequencies of the basis
#         dt: the time step of the xi process
#         length: the length of the xi process
#         cutoff: the cutoff of the kernel
#     returns
#         xi_cos_wt: shape (nbasis, nstep)
#         xi_sin_wt: shape (nbasis, nstep)
#     '''
#     ## generate the white noise array
#     nstep = int(length / dt)
#     white_noise = np.random.normal(0, dt**0.5, nstep)  
#     ### generate the xi_process
#     xi_cos_wt = []
#     xi_sin_wt = []
#     for w in ws:
#         xi_cos, xi_sin = get_xi_process(white_noise, tau, w, dt, cutoff)
#         xi_cos_wt.append(xi_cos)
#         xi_sin_wt.append(xi_sin)    
#     return np.array(xi_cos_wt) , np.array(xi_sin_wt)
 

