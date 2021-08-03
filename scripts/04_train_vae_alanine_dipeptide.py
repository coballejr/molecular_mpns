import numpy as np 
import mdtraj as md
import mdshare
from molecular_mpns.vae import VAE, VAEloss
from molecular_mpns.data import AlanineDipeptideGraph
from torch_geometric.data import DataLoader
from molecular_mpns.training_utils import *
import torch
from torch.optim.lr_scheduler import ExponentialLR
import os

figs_dir = '/afs/crc.nd.edu/user/c/coballe/molecular_mpns/figs/08032021'

# load data
xtc_filename = mdshare.fetch('alanine-dipeptide-0-250ns-nowater.xtc')
pdb_filename = mdshare.fetch('alanine-dipeptide-nowater.pdb')

traj = md.load(xtc_filename, top = pdb_filename)
traj = traj.superpose(traj)

# create graphs
batch_size = 32
z = [atom.element.atomic_number for atom in traj.topology.atoms]
G = [AlanineDipeptideGraph(z = torch.tensor(z).long(),pos = torch.tensor(xyz)) for xyz in traj.xyz]

# build model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kwargs = {'num_spherical': 16, 'num_radial': 6, 'hidden_channels': 128, 'num_bilinear': 4, 'out_channels': 64, 'num_layers': 2, 'num_blocks': 2,
         'num_enc_lins': 4, 'num_dec_lins': 4, 'enc_act': torch.tanh, 'dec_act': torch.tanh, 'dec_hidden_channels': 128, 'dec_var_const': True,
         'cutoff': 0.5, 'zdim': 2}

mod = VAE(**kwargs)
mod = mod.to(device)

opt = torch.optim.Adam(mod.parameters(),lr = 1e-4, weight_decay = 1e-5)
sched = ExponentialLR(opt, gamma = 0.995)

# training loop
os.chdir(figs_dir)
epochs, batch_size = 30, 32
subset_size, train_prop  = 10000, 0.5

train, test = train_test_split(data = G,subset_size = subset_size, train_prop = train_prop)
traj_xyz_ref = np.array([mol.pos.cpu().numpy() for mol in train])

for ep in range(epochs):
    ep_loss = 0
    
    # downsample training set
    np.random.seed(42)
    random_idx = np.random.choice(len(train),len(train), replace = False)
    G_epoch = [train[i] for i in random_idx]
    loader = DataLoader(G_epoch,batch_size = batch_size)
    
    for G_batch in loader:
        G_batch.to(device)
        
        # forward pass
        recon_batch, mu_enc, logvar_enc = mod(G_batch)
        mu_dec = recon_batch[0]
        logvar_dec = recon_batch[1]
        
        # compute loss
        loss = VAEloss(mu_dec,logvar_dec,G_batch,mu_enc,logvar_enc,L=1)
        
        # back prop
        loss.backward()
        opt.step()
        opt.zero_grad()
        ep_loss += loss.item()
    
    if (ep+1) % 1 == 0:
        
        # ancestral sampling
        traj_xyz_ancestral = ancestral_sample(mod = mod , T = subset_size, dim_z = kwargs['zdim'],device = device, sample = True)
    
        print("Epoch " + str(ep + 1) + " Loss: " + str(ep_loss))
        z0 = torch.randn((1,kwargs['zdim'])).to(device)
        x0 = train[0].pos
        traj_xyz_mwg = np.zeros((subset_size,22,3))

        for t in range(subset_size):
            z0, x0 = mod.mwg_sample(z0, x0, atomic_numbers = z, sample = True)
            traj_xyz_mwg[t,:,:] = x0.cpu().numpy()
        dih_filename, rog_filename = 'dihedrals'+str(ep)+'.png', 'rogs'+str(ep)+'.png'
        dihedrals_ref, dihedrals_ancestral, dihedrals_mwg = compare_dihedrals(traj_xyz_ref, traj_xyz_ancestral, traj_xyz_mwg ,topology = traj.topology, filename = dih_filename)
        rogs_ref, rogs_ancestral, rogs_mwg = compare_rogs(traj_xyz_ref, traj_xyz_ancestral, traj_xyz_mwg ,topology = traj.topology, filename = rog_filename)

model_path = '/afs/crc.nd.edu/user/c/coballe/molecular_mpns/models/vae04.pt'        
torch.save(mod.state_dict(), model_path)
