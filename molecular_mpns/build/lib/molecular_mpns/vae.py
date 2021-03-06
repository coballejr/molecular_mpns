from torch_geometric.nn.models.dimenet import BesselBasisLayer, SphericalBasisLayer, EmbeddingBlock
from torch_sparse import SparseTensor
from torch_geometric.nn.acts import swish
from torch_geometric.nn import radius_graph
import torch
from torch.nn import Linear
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter
from torch_geometric.data import DataLoader
from molecular_mpns.data import AlanineDipeptideGraph
import numpy as np


    
class Interaction(torch.nn.Module):
    def __init__(self, hidden_channels, num_bilinear, num_spherical,
                 num_radial, act=swish):
        super(Interaction, self).__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lin_sbf = Linear(num_spherical * num_radial, num_bilinear,
                              bias=False)

        # Dense transformations of input messages.
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)

        self.W = torch.nn.Parameter(
            torch.Tensor(hidden_channels, num_bilinear, hidden_channels))

        self.lin = Linear(hidden_channels, hidden_channels)
              
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf.weight, scale=2.0)
        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)
        self.W.data.normal_(mean=0, std=2 / self.W.size(0))
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        rbf = self.lin_rbf(rbf)
        sbf = self.lin_sbf(sbf)

        x_ji = self.act(self.lin_ji(x))
        x_kj = self.act(self.lin_kj(x))
        x_kj = x_kj * rbf
        x_kj = torch.einsum('wj,wl,ijl->wi', sbf, x_kj[idx_kj], self.W)
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))

        h = x_ji + x_kj
        h = self.act(self.lin(h))
        

        return h
    
class OutputBlock(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, out_channels, num_layers,
                 act=swish):
        super(OutputBlock, self).__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lin = Linear(hidden_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)

    def forward(self, x, rbf, i, num_nodes=None):
        x = self.lin_rbf(rbf) * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        for lin in self.lins:
            x = self.act(lin(x))
        return self.lin(x)

class MPN(torch.nn.Module):
    
    def __init__(self,num_spherical, num_radial, hidden_channels, num_bilinear, out_channels, num_layers, num_blocks = 1, cutoff=5.0,
                 envelope_exponent=5, act = swish):
        super(MPN, self).__init__()
        self.cutoff = cutoff
        self.num_blocks = num_blocks
        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff,
                                       envelope_exponent)

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act)
        
        self.output_blocks = torch.nn.ModuleList([OutputBlock(num_radial, hidden_channels, out_channels, num_layers) for _ in range(num_blocks + 1)])
        self.interaction_blocks = torch.nn.ModuleList([Interaction(hidden_channels, num_bilinear, num_spherical, num_radial) for _ in range(num_blocks)])
        
        
    def triplets(self, edge_index, num_nodes):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji
    
    def forward(self, z, pos, batch=None):
        """"""
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_ki = pos[idx_j] - pos_i, pos[idx_k] - pos_i
        a = (pos_ji * pos_ki).sum(dim=-1)
        b = torch.cross(pos_ji, pos_ki).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)
        
        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i)

        return P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)

    
class VAE(torch.nn.Module):
    
    def __init__(self, num_spherical, num_radial, hidden_channels, num_bilinear, out_channels, num_layers, num_blocks ,num_enc_lins = 4, num_dec_lins = 4, enc_act = torch.tanh, dec_act = torch.tanh, dec_hidden_channels = 256 ,dec_var_const = False, cutoff=5.0, zdim=2, xdim = 3*22,training = True):
        
        super(VAE,self).__init__()
        
        self.training = training
        self.dec_var_const = dec_var_const
        
        # encoder
        self.mpn = MPN(num_spherical, num_radial, hidden_channels, num_bilinear, out_channels, num_layers, num_blocks, cutoff=5.0)
        
        self.enc_act = enc_act
        self.enc_lins = torch.nn.ModuleList()
        for _ in range(num_enc_lins):
            self.enc_lins.append(torch.nn.Linear(out_channels, out_channels))
        
        self.enc_mu = torch.nn.Linear(out_channels,zdim)
        self.enc_logvar = torch.nn.Linear(out_channels,zdim)
    
        # decoder
        self.dec_act = dec_act
        self.dec_lins = torch.nn.ModuleList()
        for l in range(num_dec_lins):
            if l == 0:
                self.dec_lins.append(torch.nn.Linear(zdim,dec_hidden_channels))
            else:
                self.dec_lins.append(torch.nn.Linear(dec_hidden_channels, dec_hidden_channels))
        self.dec_logvar = torch.nn.Parameter(torch.zeros(xdim), requires_grad=True) if dec_var_const else torch.nn.Linear(dec_hidden_channels, xdim)
        self.dec_mu = torch.nn.Linear(dec_hidden_channels ,xdim)
        
    def encode(self,G_batch):
        
        h = self.mpn(G_batch.z,G_batch.pos,G_batch.batch)
        
        for lin in self.enc_lins:
            h = self.enc_act(lin(h))
        
        mu_enc = self.enc_mu(h)
        logvar_enc = self.enc_logvar(h)
        
        return mu_enc, logvar_enc
    
    def reparameterize(self,mu_enc,mu_logvar):
        
        sigma = torch.exp(0.5*mu_logvar)
        eps = torch.randn_like(sigma)
        z = mu_enc + (sigma*eps)
        
        return z if self.training else mu_enc
    
    def decode(self, h):
        
        for lin in self.dec_lins:
            h = self.dec_act(lin(h))
        
        mu_dec = self.dec_mu(h)
        
        if self.dec_var_const:
            batch_size = mu_dec.shape[0]
            logvar_dec = self.dec_logvar.repeat(batch_size, 1)
        else:
            logvar_dec = self.dec_logvar(h)
        
        return mu_dec, logvar_dec
    
    def forward(self, G_batch):
        mu_enc, logvar_enc = self.encode(G_batch)
        z = self.reparameterize(mu_enc, logvar_enc)
        return self.decode(z), mu_enc, logvar_enc
    
    def mwg_sample(self, z0, x0, atomic_numbers, sample = True):
        G_batch = [AlanineDipeptideGraph(z = torch.tensor(atomic_numbers).long(), pos = x0)]
        loader = DataLoader(G_batch, batch_size = 1)

        # draw proposal z_t
        for g in loader:
            g = g.to(self.enc_lins[0].weight.device)
            with torch.no_grad():
                mu_x0, logvar_x0 = self.encode(g)
                var_x0 = torch.exp(0.5*logvar_x0)
                znext = self.reparameterize(mu_x0, logvar_x0) if sample else mu_x0
        
        # compute params for p(x | z)
        with torch.no_grad():
            mu_z0, logvar_z0 = self.decode(z0)
            var_z0 = torch.exp(0.5*logvar_z0)
            mu_znext, logvar_znext = self.decode(znext)
            var_znext = torch.exp(0.5*logvar_znext)
    
        # compute p(x_{t-1} | z_t) and p(x_{t-1} | z_{t-1})
        p_z0 = torch.exp(-0.5*(((x0.flatten().to(self.enc_lins[0].weight.device) - mu_z0)*(1/var_z0)).pow(2).sum()))
        p_znext = torch.exp(-0.5*(((x0.flatten().to(self.enc_lins[0].weight.device) - mu_znext)*(1/var_znext)).pow(2).sum()))
            
        # compute p(z_t) and p(z_{t-1})
        pri_z0 = torch.exp(-0.5*(z0.pow(2).sum()))
        pri_znext = torch.exp(-0.5*(znext.pow(2).sum()))

        # compute q(z_t | x_{t-1}) and q(z_{t-1} | x_{t-1})
        q_z0 = torch.exp(-0.5*(((z0 - mu_x0)*(1/var_x0)).pow(2).sum()))
        q_znext = torch.exp(-0.5*(((znext - mu_x0)*(1/var_x0)).pow(2).sum()))

        # compute acceptance ratio
        ar = ((p_znext/p_z0)*(pri_znext/pri_z0)*(q_z0 / q_znext)).item()

        # update z0 and x0
        u = np.random.rand()
        accept = u < ar
        
        z_t = znext if accept else z0
        
        with torch.no_grad():
            mu_zt, logvar_zt = self.decode(z_t)
            x_t = self.reparameterize(mu_zt, logvar_zt) if sample else mu_zt
        x_t = x_t.reshape((22,3))
        
        return z_t, x_t
    
def VAEloss(mu_dec,logvar_dec,G_batch,mu_enc,logvar_enc,L):
    
    # recon loss for p(x | z)
    data = G_batch.pos.view(mu_dec.shape[0],mu_dec.shape[1])
    pointwiseMSEloss = 0.5*torch.nn.functional.mse_loss(mu_dec,data,reduction = 'none')
    sigsq = torch.exp(logvar_dec)
    weight = 1/sigsq
    pointwiseWeightedMSEloss = pointwiseMSEloss*weight
    WeightedMSEloss = pointwiseWeightedMSEloss.sum()
    
    logvarobjective = 0.5 * logvar_dec.sum() # scaling factor term for p(x|z)
    
    # KLD loss for q(z | x)
    KLD = -0.5 * torch.sum(1 + logvar_enc - mu_enc**2 - torch.exp(logvar_enc))
    
    loss = (KLD + (1/L)*WeightedMSEloss + logvarobjective)
    
    return loss

