import torch
from torch_geometric.nn import MessagePassing
from molecular_mpns.utils import _cdist, _dist_mat_to_edge_attr


class EdgeFeature(torch.nn.Module):
    '''
    Module to compute edge features.
    '''
    
    def __init__(self,n_rbf,rbf_range,gamma,h_dim):
        '''
        Parameters
        ----------
        n_rbf : number of radial basis centers, int.
        rbf_range : array-like, [b,e] where b and e are first and last radial basis centers resp.
        gamma : gamma param in rbf function, float.
        h_dim : number of hidden units in dense layers, int.
        '''
        
        super(EdgeFeature,self).__init__()
        
        self.register_buffer('centers',torch.linspace(rbf_range[0],rbf_range[1],n_rbf))
        self.register_buffer('gamma',torch.tensor(gamma))
        self.dense1 = torch.nn.Linear(n_rbf,h_dim)
        self.dense2 = torch.nn.Linear(h_dim,h_dim)
        
    def forward(self,edge_attr):
        '''
        Parameters
        ----------
        edge_attr : tensor of edge attributes (e.g., dists), shape = (n_edges, attr_dim).

        Returns
        -------
        a : edge features, tensor, shape = (n_edges,h_dim).
        '''
        
        center_dists = self.centers-edge_attr
        a = torch.exp(-self.gamma*(center_dists**2))
        a = self.dense1(a)
        a = torch.nn.functional.silu(a)
        a = self.dense2(a)
        a = torch.nn.functional.silu(a)
        
        return a
    
class CFConv(MessagePassing):
    '''
    Module for continuous filter convoloution.
    '''
    
    def __init__(self,n_rbf,rbf_range,gamma,h_dim):
        '''
        Parameters
        ----------
        n_rbf : number of radial basis centers, int.
        rbf_range : array-like, [b,e] where b and e are first and last radial basis centers resp.
        gamma : gamma param in rbf function, float.
        h_dim : number of hidden units in dense layers, int.
        '''
        
        super(CFConv,self).__init__(aggr = 'add')
        self.edge_features = EdgeFeature(n_rbf,rbf_range,gamma,h_dim)
        
    def forward(self,edge_index,edge_attr,x):
        ''' Precomputes edge features and relays information for message-passing.
        
        Parameters
        ----------
        edge_index : long tensor, shape = (2,n_edges), [source_nodes,target_nodes] .
        edge_attr : tensor of edge attributes (e.g., dists), shape = (n_edges, attr_dim).
        x : node features, tensor, shape = (n_nodes,h_dim)
        '''
        a = self.edge_features(edge_attr)
        return self.propagate(edge_index=edge_index,a=a,x=x)
    
    def message(self,x_j,a,flow = 'source_to_target'):
        ''' Message-passing step.
        

        Parameters
        ----------
        x_j : source nodes, tensor, shape = (n_nodes*n_nodes,h_dim).
        a : edge features, tensor, shape = (n_edges,h_dim).

        Returns
        -------
        messages grouped by source node, tensor, shape = (n_nodes,h_dim).

        '''
        return a*x_j
    
class Interaction(torch.nn.Module):
    '''
    Module for interaction block.
    '''
    
    def __init__(self,n_rbf,rbf_range,gamma,dim):
        '''
        Parameters
        ----------
        n_rbf : number of radial basis centers, int.
        rbf_range : array-like, [b,e] where b and e are first and last radial basis centers resp.
        gamma : gamma param in rbf function, float.
        dim : number of hidden units in dense layers, int.

        Returns
        -------
        None.

        '''
        super(Interaction,self).__init__()
        
        self.atomwise1 = torch.nn.Linear(dim,dim)
        self.cfconv = CFConv(n_rbf,rbf_range,gamma,dim)
        self.atomwise2 = torch.nn.Linear(dim,dim)
        self.atomwise3 = torch.nn.Linear(dim,dim)
        
    def forward(self,edge_index,edge_attr,x):
        
        h = self.atomwise1(x)
        h = self.cfconv(edge_index,edge_attr,h)
        h = self.atomwise2(h)
        h = torch.nn.functional.silu(h)
        h = self.atomwise3(h)
        return x + h
    
class ProtoNet(torch.nn.Module):
    
    def __init__(self,emb_dim,intermediate_dim,n_rbf,rbf_range,gamma,out_dim):
        super(ProtoNet,self).__init__()
        
        self.emb_dim = emb_dim
        self.embedding = torch.nn.Embedding(1,emb_dim)
        
        self.interaction1 = Interaction(n_rbf,rbf_range,gamma,emb_dim)
        self.interaction2 = Interaction(n_rbf,rbf_range,gamma,emb_dim)
        self.interaction3 = Interaction(n_rbf,rbf_range,gamma,emb_dim)
        
        self.atomwise1 = torch.nn.Linear(emb_dim,intermediate_dim)
        self.atomwise2 = torch.nn.Linear(intermediate_dim,out_dim)
        
    def forward(self,mol_graph):
        '''
    
        Parameters
        ----------
        mol_graph : MarkovMolGraph object.

        Returns
        -------
        V : molecular potential, tensor, shape = (1,).

        '''
        
        edge_index,x,r_current = mol_graph.edge_index,mol_graph.Z,mol_graph.r_current
        edge_attr = _cdist(r_current)
        edge_attr = _dist_mat_to_edge_attr(edge_attr)
        
        h = self.embedding(x)
        h = h.view(x.shape[0],self.emb_dim)
        h = self.interaction1(edge_index,edge_attr,h)
        h = self.interaction2(edge_index,edge_attr,h)
        h = self.interaction3(edge_index,edge_attr,h)
        h = self.atomwise1(h)
        h = self.atomwise2(h)
        
        return h
    
class OUNet(torch.nn.Module):
    
    def __init__(self,hdim):
        
        super(OUNet,self).__init__()
        self.lin1 = torch.nn.Linear(1,hdim)
        self.lin2 = torch.nn.Linear(hdim,hdim)
        self.lin3 = torch.nn.Linear(hdim,hdim)
        self.lin4 = torch.nn.Linear(hdim,hdim)
        self.lin5 = torch.nn.Linear(hdim,hdim)
        self.out = torch.nn.Linear(hdim,1)
        
    def forward(self,x):
        h = self.lin1(x)
        h = torch.nn.functional.silu(h)
        h = self.lin2(h)
        h = torch.nn.functional.silu(h)
        h = self.lin3(h)
        h = torch.nn.functional.silu(h)
        h = self.lin4(h)
        h = torch.nn.functional.silu(h)
        h = self.lin5(h)
        h = torch.nn.functional.silu(h)
        y = self.out(h)
        return y
    
class DblWellXNet(torch.nn.Module):
    
    def __init__(self,hdim):
        
        super(DblWellXNet,self).__init__()
        self.lin1 = torch.nn.Linear(2,hdim)
        self.lin2 = torch.nn.Linear(hdim,hdim)
        self.lin3 = torch.nn.Linear(hdim,hdim)
        self.lin4 = torch.nn.Linear(hdim,hdim)
        self.lin5 = torch.nn.Linear(hdim,hdim)
        
        self.bn1 = torch.nn.BatchNorm1d(hdim)
        self.bn2 = torch.nn.BatchNorm1d(hdim)
        self.bn3 = torch.nn.BatchNorm1d(hdim)
        self.bn4 = torch.nn.BatchNorm1d(hdim)
        self.bn5 = torch.nn.BatchNorm1d(hdim)
        
        self.out = torch.nn.Linear(hdim,2)
        
    def forward(self,x):
        h = self.lin1(x)
        #h = self.bn1(h)
        h = torch.nn.functional.silu(h)
        h = self.lin2(h)
        #h = self.bn2(h)
        h = torch.nn.functional.silu(h)
        h = self.lin3(h)
        #h = self.bn3(h)
        h = torch.nn.functional.silu(h)
        h = self.lin4(h)
        #h = self.bn4(h)
        h = torch.nn.functional.silu(h)
        h = self.lin5(h)
        #h = self.bn5(h)
        h = torch.nn.functional.silu(h)
        y = self.out(h)
        return y
    

    

        