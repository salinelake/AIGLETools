import numpy as np
import torch as th
from scipy.ndimage import convolve1d

def th2np(x):
    return x.detach().cpu().numpy()

def np2th(x):
    dev = 'cuda' if th.cuda.is_available() else 'cpu'
    return th.tensor(x, dtype=th.float, device=dev)

def Corr_t(x,y, l ):
    """"x(0)y(t)"""
    corr = [  (x*y).mean(0)] + [ (x[:-iT] * y[iT:]).mean(0)  for iT in range(1, l)]
    if type(x)==np.ndarray:
        return np.array(corr)
    else:
        return th.stack(corr)

def fpt(traj, start, end, epsilon = 0.05):
    def d(a,b):
        if type(a) is np.ndarray or th.tensor:
            return ((a-b)**2).sum()**0.5
        else:
            return np.abs(a-b)
    start_list = []
    fpt_list = []
    end_list = []
    for idx, x in enumerate(traj):
        if d(x, start) < epsilon:
            start_list.append( idx * 1 )
        if len(start_list) > 0 and d(x, end) < epsilon:
            fpt = idx - np.array(start_list)
            fpt_list.append(fpt)
            start_list = []
            end_list.append(idx * 1 )
    fpt_list = np.concatenate(fpt_list)
    return fpt_list
 
# def moving_average_half_gaussian(a, sigma=25, truncate=3.0):
#     '''
#     smooth over the first axis with a half gaussian kernel
#     '''
#     fsize = int(truncate * np.ceil(sigma))
#     weights = [ np.exp(-x**2/2.0/sigma**2) for x in range(fsize) ]
#     throw = fsize//2 + 1
#     weights = np.array(weights)
#     weights = weights / weights.sum()
#     ret = convolve1d(a, weights, axis=0, origin=1 )
#     return ret[throw:-throw]

def moving_average_half_gaussian_torch(a, sigma=25, truncate=3.0):
    '''
    smooth over the first axis with a half gaussian kernel
    Args:
        a: tensor, (nframes, ndim)
        sigma: float, the standard deviation of the gaussian kernel
        truncate: integer, the truncation of the gaussian kernel
    '''
    fsize = int(truncate * np.ceil(sigma))
    weights = [ np.exp(-x**2/2.0/sigma**2) for x in range(fsize) ]
    weights = np.array(weights)
    weights = weights / weights.sum()
    weights = th.tensor(weights, dtype=a.dtype, device=a.device)[None,None,:] # (1,1, kW) #->  conv1d: (out_ch, in_ch/groups, kW)
    filtered_a = th.transpose(a, 0, 1)[:,None,:]  # (ndim, 1, nframes)  ->  conv1d: (minibatch, in_ch, iW)
    filtered_a = th.nn.functional.conv1d(filtered_a, weights,  padding='valid')  # (ndim, 1, nframes - *)
    filtered_a = filtered_a.squeeze(1).transpose(0,1)  # (nframes - *, ndim)
    return filtered_a

def get_exact_memory_kernel(_corr_vtv0, _corr_qtv0, kernel_length, dt):
    """
    get the exact result of the memory kernel through direct matrix inversion
    """
    ## check type
    if th.is_tensor(_corr_vtv0):
        corr_vtv0 = _corr_vtv0
    else:
        if type(_corr_vtv0) is np.ndarray:
            corr_vtvo = np2th(_corr_vtv0)
        else:
            raise TypeError('corr_vtvo should be an array or a tensor')
    if th.is_tensor(_corr_qtv0):
        corr_qtv0 = _corr_qtv0
    else:
        if type(_corr_qtv0) is np.ndarray:
            corr_qtvo = np2th(_corr_qtv0)
        else:
            raise TypeError('corr_qtvo should be an array or a tensor')
    ## get <v(t+0.5)v(0)> for integer t       
    corr_vvplus = (corr_vtv0[1:] + corr_vtv0[:-1])/2
    mat_cvv = np2th(np.zeros((kernel_length-1, kernel_length-1)))
    for ii in range(mat_cvv.shape[0]):
        mat_cvv[ii, :ii+1] = th.flip(corr_vvplus[ :ii+1 ], [0])
        # for jj in range(ii+1):
        #     mat_cvv[ii, jj] = corr_vvplus[ ii-jj ]
    ## least square solution to <q(t)v(0)>=\int K(s) v(t-s)v(0)ds    
    fit_b = corr_qtv0[1:kernel_length] 
    fit_A = mat_cvv * dt
    fit_t = np2th(dt*(np.arange(kernel_length-1)+0.5))
    lstsq_results = th.linalg.lstsq(fit_A, fit_b, rcond=None)
    ref_mem_kernel = lstsq_results.solution
    return ref_mem_kernel, fit_t, fit_A, fit_b



def binned_correlation(x1, x2, shift, nbins, bin_indices):
    """
    The conditional correlation: f(r, n) = <x1(i)x2(i-n)>*delta(x1(i)-r)
    Args:
        shift: n, integer
        x1: array or tensor, (npoints)
        x2: array or tensor, (npoints)
        bin_indices: array or tensor, (npoints). The bin index for each element of x1
    Returns:
        binned_corr: f(r, shift) for r value given by the bins, shape=(nbins)
    """
    assert x1.shape == x2.shape, 'the two array/tensor should have the same shape'
    assert x1.shape == bin_indices.shape, 'x1 and its bin indices should have the same shape'
    if shift == 0:
        corr = x1*x2
    else:
        corr = x1[shift:] * x2[:-shift]
    ## initialize
    binned_corr = np.zeros(nbins)
    if th.is_tensor(x1):
        binned_corr = np2th(binned_corr)
    ## fill up
    for ii in range( nbins):
        binned_corr[ii] = corr[bin_indices[shift:]==(ii+1)].mean()
    return binned_corr