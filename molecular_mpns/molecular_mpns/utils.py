import torch


def _cdist(r,eps = 1e-7):
    
    ''' Custom cdist function to use for second-order gradients.
    
    Parameters
    ----------
    r : tensor of atomic coordinates, shape = (n_atoms,coord_dim).
    eps : prevents zeros in square root gradient. The default is 1e-7.

    Returns
    -------
    upper triangular tensor t, shape = (n_atoms,n_atoms), t[i][j] = dist(r[i],[r[j]).

    '''
    
    disps = r - r.reshape((r.shape[0],1,r.shape[1]))
    D2 = torch.einsum('vnd,vnd -> vn',disps,disps)+eps
    return torch.triu(torch.sqrt(D2))


def _dist_mat_to_edge_attr(m):
    ''' Reshapes output of _cdist into edge features.

    Parameters
    ----------
    m : upper triangular tensor, shape = (n_atoms,n_atoms), output of _cdist.

    Returns
    -------
    edge_attr : tensor, shape = (n_atoms*n_atoms,1).
    '''
    
    edge_attr = m+m.t()
    edge_attr = edge_attr.view(m.shape[0]*m.shape[0],1)
    
    return edge_attr

