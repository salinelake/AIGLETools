import numpy as np
import torch as th
from .utilities import th2np
class interpolated_fes_2d:
    def __init__(self, fes_table, x0, y0, dx, dy, pbc, use_torch=False):
        '''
        Args:
            fes_table: 2d array, the free energy surface
            x0, y0: the origin of the free energy surface
            dx, dy: the grid size of the free energy surface
        '''
        self.fes = fes_table
        self.use_torch = use_torch
        self.nx, self.ny = fes_table.shape
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.dy = dy
        self.lx = dx * (self.nx-1)
        self.ly = dy * (self.ny-1)
        if pbc is True:
            if np.abs(self.fes[0] - self.fes[-1]).max() > 1e-6:
                raise ValueError("Periodic boundary condition is not satisfied, the free energy at the first and last row are different")
            if np.abs(self.fes[:,0] - self.fes[:,-1]).max() > 1e-6:
                raise ValueError("Periodic boundary condition is not satisfied, the free energy at the first and last column are different")
            # calculate the gradient of the free energy surface
            self.grad_x = ( np.roll(fes_table,-1,0) - np.roll(fes_table, 1, 0 )) / 2 / dx
            self.grad_y = ( np.roll(fes_table,-1,1) - np.roll(fes_table, 1, 1 )) / 2 / dy
        else:
            self.grad_x = np.gradient(fes_table, [dx,dy], axis=0)
            self.grad_y = np.gradient(fes_table, [dx,dy], axis=1)

    def scaled_position(self, x, y):
        '''
        Always impose minimum image convention, no matter if the system is periodic or not
        Returns:
            relative coordinates in the cell with respect to origin (x0,y0)
        '''
        x = x - self.lx * np.floor((x-self.x0)/self.lx)
        y = y - self.ly * np.floor((y-self.y0)/self.ly)
        x = (x - self.x0) / self.dx
        y = (y - self.y0) / self.dy
        return x, y

    def interpolate(self, m, shift_x, shift_y):
        '''
        Interpolate the value of a 2x2 matrix
        Args:
            m: 2x2 array, the matrix to be interpolated
            shift_x, shift_y: the shift of the point to the base point
        '''
        f0 = (1-shift_x) * m[0, 0] + shift_x * m[1,0]
        f1 = (1-shift_x) * m[0, 1] + shift_x * m[1,1]
        result = (1-shift_y)* f0 + shift_y * f1
        return result

    def calc_energy_single(self, r):
        '''
        Calculate the free energy at a given position r
        '''
        if self.use_torch:
            pos = th2np(r)
        else:
            pos = r
        # get the position in the primary cell
        x, y = self.scaled_position(pos[0], pos[1])
        base_idx_x = int(np.floor(x))
        base_idx_y = int(np.floor(y))
        shift_x = x - base_idx_x
        shift_y = y - base_idx_y
        ## calculate the free energy
        submat = self.fes[base_idx_x:base_idx_x+2, base_idx_y:base_idx_y+2]
        energy = self.interpolate(submat, shift_x, shift_y)
        if self.use_torch:
            return th.tensor(energy, device=r.device, dtype=r.dtype)
        else:
            return energy

    def calc_force_single(self, r):
        '''
        Calculate the force at a given position r
        Args:
            r: 1d numpy array of shape (2) , the position
        '''
        # get the position in the primary cell
        x, y = self.scaled_position(r[0], r[1])
        base_idx_x = int(np.floor(x))
        base_idx_y = int(np.floor(y))
        shift_x = x - base_idx_x
        shift_y = y - base_idx_y
        ## calculate the force
        grad_x_submat = self.grad_x[base_idx_x:base_idx_x+2, base_idx_y:base_idx_y+2]
        fx = - self.interpolate(grad_x_submat, shift_x, shift_y)
        grad_y_submat = self.grad_y[base_idx_x:base_idx_x+2, base_idx_y:base_idx_y+2]
        fy = - self.interpolate(grad_y_submat, shift_x, shift_y)
        return np.array([fx, fy])

    def calc_force_batched(self, r):
        '''
        Calculate the force at given positions r
        Args:
            r: 2d numpy array of shape (batchsize, 2) , the positions
        '''
        # get the positions in the primary cell
        x, y = self.scaled_position(r[:,0], r[:,1])
        base_idx_x = np.floor(x).astype(int)
        base_idx_y = np.floor(y).astype(int)
        shift_x = x - base_idx_x
        shift_y = y - base_idx_y
        ## calculate the force
        grad_x_m00 = self.grad_x[base_idx_x, base_idx_y]
        grad_x_m01 = self.grad_x[base_idx_x, base_idx_y+1]
        grad_x_m10 = self.grad_x[base_idx_x+1, base_idx_y]
        grad_x_m11 = self.grad_x[base_idx_x+1, base_idx_y+1]
        grad_x_submat = np.stack([grad_x_m00, grad_x_m01, grad_x_m10, grad_x_m11], axis=0).reshape(2,2,-1)
        grad_y_m00 = self.grad_y[base_idx_x, base_idx_y]
        grad_y_m01 = self.grad_y[base_idx_x, base_idx_y+1]
        grad_y_m10 = self.grad_y[base_idx_x+1, base_idx_y]
        grad_y_m11 = self.grad_y[base_idx_x+1, base_idx_y+1]
        grad_y_submat = np.stack([grad_y_m00, grad_y_m01, grad_y_m10, grad_y_m11], axis=0).reshape(2,2,-1)
        fx = - self.interpolate(grad_x_submat, shift_x, shift_y)
        fy = - self.interpolate(grad_y_submat, shift_x, shift_y)
        result = np.stack([fx, fy], axis=-1)
        return result

    def calc_force(self, r):
        '''
        Calculate the force at given positions r
        Args:
            r: 1d array of shape (2) or 2d array of shape (batchsize, 2) , the positions
        '''
        if self.use_torch:
            pos = th2np(r)
        else:
            pos = r

        if pos.ndim == 1:
            force = self.calc_force_single(pos)
        else:
            force = self.calc_force_batched(pos)

        if self.use_torch:
            return th.tensor(force, device=r.device, dtype=r.dtype)
        else:
            return force

    def calc_energy(self, r):
        '''
        Calculate the free energy at given positions r
        Args:
            r: 1d array of shape (2) or 2d array of shape (batchsize, 2) , the positions
        '''
        if self.use_torch:
            pos = th2np(r)
        else:
            pos = r
        if pos.ndim == 1:
            energy = self.calc_energy_single(pos)
        else:
            energy = np.array([self.calc_energy_single(p) for p in pos])
            
        if self.use_torch:
            return th.tensor(energy, device=r.device, dtype=r.dtype)
        else:
            return energy