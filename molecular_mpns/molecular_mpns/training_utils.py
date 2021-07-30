import numpy as np 
import mdtraj as md
import torch
import matplotlib.pyplot as plt
from molecular_mpns.math import radius_of_gyration



def train_test_split(data, subset_size, train_prop, seed = 42):
    
    np.random.seed(seed)
    assert subset_size < len(data)
    random_idx = np.random.choice(len(data), subset_size, replace = False)
    
    assert train_prop < 1
    train_size = int(train_prop*subset_size)
    train_idx, test_idx = random_idx[0:train_size], random_idx[train_size:]
    
    train, test = [data[i] for i in train_idx], [data[i] for i in test_idx]
    
    return train, test


def compare_dihedrals(traj_xyz_ref, traj_xyz_ancestral, traj_xyz_mwg ,topology, bins = 40, rng = [[-np.pi,np.pi],[-np.pi, np.pi]],psi_inds = [6, 8, 14, 16], phi_inds = [4, 6, 8, 14]):
    traj_ref, traj_ancestral, traj_mwg = md.Trajectory(traj_xyz_ref, topology = topology), md.Trajectory(traj_xyz_ancestral, topology = topology), md.Trajectory(traj_xyz_mwg, topology = topology)
    dihedrals_ref, dihedrals_ancestral, dihedrals_mwg = md.compute_dihedrals(traj_ref,[psi_inds,phi_inds]), md.compute_dihedrals(traj_ancestral,[psi_inds,phi_inds]), md.compute_dihedrals(traj_mwg,[psi_inds,phi_inds])
    
    fig, ax = plt.subplots(1, 3, sharey = True, sharex = True)
    
    psi, phi = dihedrals_ref[:,1], dihedrals_ref[:,0]
    ax[0].hist2d(psi, phi, bins = bins, range = rng)
    ax[0].set_aspect('equal')
    ax[0].set_title('Reference')
    
    psi, phi = dihedrals_ancestral[:,1], dihedrals_ancestral[:,0]
    ax[1].hist2d(psi, phi, bins = bins, range = rng)
    ax[1].set_aspect('equal')
    ax[1].set_title('Ancestral')
    
    psi, phi = dihedrals_mwg[:,1], dihedrals_mwg[:,0]
    ax[2].hist2d(psi, phi, bins = bins, range = rng)
    ax[2].set_aspect('equal')
    ax[2].set_title('Met.-W.-Gibbs')
    
    plt.show()
    plt.close()
    
    return dihedrals_ref, dihedrals_ancestral, dihedrals_mwg



def compare_rogs(traj_xyz_ref, traj_xyz_anc, traj_xyz_mwg, topology, density = True, xlims = [0.2, 0.28], histtype = 'step'):
    rogs_ref = [radius_of_gyration(xyz, topology = topology) for xyz in traj_xyz_ref]
    rogs_anc = [radius_of_gyration(xyz, topology = topology) for xyz in traj_xyz_anc]
    rogs_mwg = [radius_of_gyration(xyz, topology = topology) for xyz in traj_xyz_mwg]
    
    plt.hist(rogs_ref, density = density, histtype = histtype, label = "Reference")
    plt.hist(rogs_anc, density = density, histtype = histtype, label = "Ancestral")
    plt.hist(rogs_mwg, density = density, histtype = histtype, label = "Met.-W.-Gibbs")
    
    #plt.xlim(xlims)
    plt.legend()
    plt.show()
    plt.close()
    
    return rogs_ref, rogs_anc, rogs_mwg

def ancestral_sample(mod, T, device ,dim_z = 2, n_atoms = 22):
    w = torch.randn((T, dim_z))
    with torch.no_grad():
        mu_dec, logvar_dec = mod.decode(w.to(device))
        x_recon = mod.reparameterize(mu_dec, logvar_dec)
    
    x_recon = x_recon.cpu().numpy()
    x_recon = x_recon.reshape((T, n_atoms ,3))
    return x_recon
