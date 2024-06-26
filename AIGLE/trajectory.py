import numpy as np
import torch as th
from .utilities import np2th, th2np, moving_average_half_gaussian_torch
from .decay_fourier import *

class Trajectory:
    '''
    The class for storing and analyzing trajectory of collective variables.
    '''
    def __init__(self, traj, dt, kbT=1):
        '''
        Args:
            traj: 2d numpy array of shape (nframes, ndim) , the trajectory
            ndim: int, the number of dimensions
            dt: float, the time step
        '''
        if traj.ndim == 2 or traj.ndim == 3:
            pass
        else:
            raise ValueError('traj should be 2d or 3d array of shape (nframes,  *)')
        if th.is_tensor(traj):
            self.traj = traj
        else:
            self.traj = np2th(traj)
        self.naxis = traj.ndim
        self.dt = dt
        self.kbT = kbT
        self.nframes = traj.shape[0]
        self.ndim = traj.shape[1]
        ##
        self.kinetics_dict = {
            'x': None,
            'v': None,
            'v_half_grid': None,  ## advanced by half time step
            'a': None,
            'f': None,
        }
        ##
        self.transform_matrix = None
        self.v2_avg = None
        self.mass = None
        self.calc = None
        self.vacf = None
        self.Ivcf = None
        self.msd = None
        self.diffusivity_half_grid = None
        self.corr_dict = {}

    def set_calculator(self, calc):
        '''
        Set the calculator for computing energy and forces.
        calc.calc_force( traj: (nframes, ndim) ) should return the force
        calc.calc_energy( traj: (nframes, ndim) ) should return the energy
        Args:
            calc: object, the calculator
        '''
        self.calc = calc
        return

    def transform(self, x ):
        '''
        Transform a 2D/3D tensor with an orthogonal rotation matrix.
        Args:
            x: tensor, (nframes, ndim, *)
            eigM: tensor, (ndim, ndim), the orthogonal rotation matrix
        '''
        if x.ndim == 2:
            return th.matmul(x, self.transform_matrix)
        else:
            return th.matmul(x.transpose(1,2), self.transform_matrix).transpose(1,2)

    def process_kinetics(self, smooth_window=None, transform=False):
        '''
        Compute and store position, velocity, acceleration on the same grid with finite difference.
        Compute also the effective mass with equipartition theorem.
        Args:
            transform: bool, whether to transform the trajectory with an orthogonal rotation matrix, 
                        to remove the instantaneous velocity correlation between different components.
        '''
        if smooth_window is None:
            traj = self.traj
        else:
            traj = moving_average_half_gaussian_torch(self.traj, sigma=smooth_window)
        v = (traj[1:] - traj[:-1]) / self.dt 
        if self.calc is not None:
            f = self.calc.calc_force(traj[1:-1])
            self.kinetics_dict['f'] = f
        if transform:
            # remove the instantaneous velocity correlation between different components
            cross_vv = (v.unsqueeze(2) * v.unsqueeze(1)).mean(0)
            if self.naxis == 3:
                cross_vv = cross_vv.mean(-1)
            eig, eigM = th.linalg.eigh(cross_vv)
            self.transform_matrix = eigM
            v = self.transform(v)
            self.kinetics_dict['x'] = self.transform(traj[1:-1])
            if self.calc is not None:
                self.kinetics_dict['f'] = self.transform(f)
        else:
            self.kinetics_dict['x'] = traj[1:-1].clone()
        self.kinetics_dict['v_half_grid'] = v[1:]
        self.kinetics_dict['v'] = (v[1:] + v[:-1]) / 2
        self.kinetics_dict['a'] = (v[1:] - v[:-1]) / self.dt  
        self.v2_avg = th.mean(v**2, dim=0)
        if self.naxis == 3:
            self.v2_avg = self.v2_avg.mean(-1)
        self.mass = self.kbT / self.v2_avg
        return
    

    def compute_vacf(self, nmax=1000, skipheads=0):
        '''
        Compute velocity autocorrelation function.
        Returns:
            vacf: tensor of shape (nmax, *), the velocity autocorrelation function
        '''
        if self.kinetics_dict['v'] is None:
            raise ValueError('The kinetics is not processed yet. run process_kinetics() first.')
        v = self.kinetics_dict['v_half_grid'][skipheads:]
        # vacf = th.zeros((nmax, self.ndim), dtype=v.dtype, device=v.device)
        # vacf[0] = th.mean(v**2, dim=0)
        # for i in range(1,nmax):
        #     vacf[i] = th.mean(v[:-i] * v[i:], dim=0)
        # self.vacf = vacf
        vacf = []
        vacf.append(th.mean(v**2, dim=0))
        for i in range(1,nmax):
            vacf.append(th.mean(v[:-i] * v[i:], dim=0))
        vacf = th.stack(vacf, 0)
        self.vacf = vacf
        return vacf

    def compute_Ivcf(self, nmax=1000, skipheads=0):
        '''
        Compute impulse velocity correlation function.
        Returns:
            Ivcf: tensor of shape (nmax, *), the impulse velocity correlation function
        '''
        if self.kinetics_dict['f'] is None:
            raise ValueError('The force is not calculated yet. set a calculator by set_calc() and run process_kinetics().')
        f = self.kinetics_dict['f'][skipheads+1:]
        v = self.kinetics_dict['v_half_grid'][skipheads:-1]

        # fvcf_half_grid= th.zeros((nmax, self.ndim), dtype=v.dtype, device=v.device)
        # fvcf_half_grid[0] = th.mean(f * v, dim=0)
        # for i in range(1,nmax):
        #     fvcf_half_grid[i] = th.mean(v[:-i] * f[i:], dim=0)
        fvcf_half_grid = []
        fvcf_half_grid.append(th.mean(f * v, dim=0))
        for i in range(1,nmax):
            fvcf_half_grid.append(th.mean(v[:-i] * f[i:], dim=0))
        fvcf_half_grid = th.stack(fvcf_half_grid, 0)
        Icvf = th.cumsum(fvcf_half_grid, 0) * self.dt
        Icvf = th.cat([th.zeros_like(Icvf[0]).unsqueeze(0), Icvf], 0)  ## the first integral of <f(t)v(0)>
        Icvf = Icvf[:-1]
        self.Ivcf = Icvf
        return Icvf
 
    def compute_msd(self, nmax=1000, skipheads=0):
        '''
        Compute mean square displacement.
        Returns:
            msd: tensor of shape (nmax, *), the mean square displacement
        '''
        if self.kinetics_dict['x'] is None:
            raise ValueError('The kinetics is not processed yet. run process_kinetics() first.')
        x = self.kinetics_dict['x'][skipheads:]
        # msd = th.zeros((nmax, self.ndim), dtype=x.dtype, device=x.device)
        # msd[0] = th.zeros_like(x[0])
        # for i in range(1,nmax):
        #     msd[i] = th.mean((x[:-i] - x[i:])**2, dim=0)
        # self.msd = msd
        msd = []
        msd.append(th.zeros_like(x[0]))
        for i in range(1,nmax):
            msd.append(th.mean((x[:-i] - x[i:])**2, dim=0))
        msd = th.stack(msd, 0)
        self.msd = msd
        return msd

    def compute_diffusivity_half_grid(self, from_msd=True):
        '''
        Compute the dynamical diffusivity D(t) on half grid.
        Args:
            from_msd: bool, whether to compute the diffusivity by taking the time derivative of the mean square displacement, 
                        or by integrating the velocity autocorrelation function.
        Returns:
            diffusivity: tensor of shape (nmax-1, *), the dynamical diffusivity D(t)
        '''
        if from_msd:
            if self.msd is None:
                raise ValueError('The mean square displacement is not computed yet. run compute_msd() first.')
            msd = self.msd
            diffusivity = (msd[1:] - msd[:-1]) / (2 * self.dt)
        else:
            if self.vacf is None:
                raise ValueError('The velocity autocorrelation function is not computed yet. run compute_vacf() first.')
            vacf = self.vacf
            int_vacf = th.cumsum( (vacf[1:]+vacf[:-1])/2, 0) * self.dt
            int_vacf = th.concatenate([th.zeros_like(int_vacf[0]).unsqueeze(0), int_vacf], 0)  ## the first integral of VACF
            diffusivity = (int_vacf[1:]+int_vacf[:-1])/2
        self.diffusivity_half_grid = diffusivity
        return diffusivity
