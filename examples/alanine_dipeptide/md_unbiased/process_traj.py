import numpy as np
import mdtraj


dt = 0.004 # ps
traj = mdtraj.load('./traj_100ns.dcd', top="./alanine-dipeptide.pdb")
nframes = len(traj)
print('duration={}ns'.format(nframes * dt/1000))

phi = mdtraj.compute_phi(traj)[1].flatten() 
psi = mdtraj.compute_psi(traj)[1].flatten() 
v_phi = phi[1:] - phi[:-1]
v_phi[v_phi > np.pi] -= 2 * np.pi
v_phi[v_phi < -np.pi] += 2 * np.pi
v_phi = v_phi / dt   #1/fs

v_psi = psi[1:] - psi[:-1]
v_psi[v_psi > np.pi] -= 2 * np.pi
v_psi[v_psi < -np.pi] += 2 * np.pi
v_psi = v_psi/ dt   #1/fs

x_phi = phi[0] + np.cumsum(v_phi)*dt
x_psi = psi[0] + np.cumsum(v_psi)*dt

cv_x = np.concatenate([x_phi[:,None],x_psi[:,None]],-1)
cv_v = np.concatenate([v_phi[:,None],v_psi[:,None]],-1)
print('CV shape:', cv_x.shape)

np.save('cv_traj_100ns_DT4fs', cv_x)