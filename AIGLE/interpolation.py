import numpy as np
class interpolated_fes_2d:
    def __init__(self, fes_table, x0, y0, dx, dy, pbc):
        '''
        Args:
            fes_table: 2d array, the free energy surface
            x0, y0: the origin of the free energy surface
            dx, dy: the grid size of the free energy surface
        '''
        self.fes = free_energy_table
        self.nx, self.ny = free_energy_table.shape
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
            self.grad_x = ( np.roll(free_energy,-1,0) - np.roll(free_energy, 1, 0 )) / 2 / dx
            self.grad_y = ( np.roll(free_energy,-1,1) - np.roll(free_energy, 1, 1 )) / 2 / dy
        else:
            self.grad_x = np.gradient(free_energy_table, [dx,dy], axis=0)
            self.grad_y = np.gradient(free_energy_table, [dx,dy], axis=1)
    def mic(self, x, y):
        '''
        Minimum image convention
        '''
        x = x - self.lx * np.floor((x-x0)/self.lx)
        y = y - self.ly * np.floor((y-y0)/self.ly)
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

    def calc_energy(self, r):
        '''
        Calculate the free energy at a given position r
        '''
        # get the position in the primary cell
        x, y = self.mic(r[0], r[1])
        x = (x - x0) / self.dx
        y = (y - y0) / self.dy
        base_idx_x = int(np.floor(x))
        base_idx_y = int(np.floor(y))
        shift_x = x - base_idx_x
        shift_y = y - base_idx_y
        ## calculate the free energy
        submat = self.fes[base_idx_x:base_idx_x+2, base_idx_y:base_idx_y+2]
        return self.interpolate(submat, shift_x, shift_y)



    def calc_force(self, r):
        '''
        Calculate the force at a given position r
        '''
        # get the position in the primary cell
        x, y = self.mic(r[0], r[1])
        x = (x - x0) / self.dx
        y = (y - y0) / self.dy
        base_idx_x = int(np.floor(x))
        base_idx_y = int(np.floor(y))
        shift_x = x - base_idx_x
        shift_y = y - base_idx_y
        ## calculate the force
        grad_x_submat = self.grad_x[base_idx_x:base_idx_x+2, base_idx_y:base_idx_y+2]
        grad_y_submat = self.grad_y[base_idx_x:base_idx_x+2, base_idx_y:base_idx_y+2]
        fx = - self.interpolate(grad_x_submat, shift_x, shift_y)
        fy = - self.interpolate(grad_y_submat, shift_x, shift_y)
        return np.array([fx, fy])
 
 
