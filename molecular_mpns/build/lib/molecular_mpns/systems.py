import numpy as np
from scipy.spatial.distance import cdist

class OrnsteinUhlenbeck:
    
    def __init__(self,alpha,beta):
        ''' Ornstein-Uhlenbeck simulator.
        

        Parameters
        ----------
        alpha : float, alpha param in OH process.
        beta : float, beta param ni OH process.

        Returns
        -------
        None.

        '''
        
        self.alpha = alpha
        self.beta = beta
        
    def _gradient(self,x):
        return self.alpha*x
        
    def step(self,x,dt):
        ''' Evolve OH process by one timestep
        

        Parameters
        ----------
        x : float, current position of process.
        dt : float, length of timestep in seconds.

        Returns
        -------
        x_next : float, next position of process.

        '''
        x_next = x - self.alpha*x*dt + np.sqrt((2*dt)/self.beta)*np.random.randn(1)
        
        return x_next
    
    def trajectory(self,x0,dt,nsteps):
        '''
        

        Parameters
        ----------
        x0 : float, initial position.
        dt : float, length of timestep in seconds.
        nsteps : int, number of steps for trajectory.

        Returns
        -------
        traj : array, shape = (1,n_steps+1).

        '''
        traj = np.zeros(nsteps+1)
        x = x0
        traj[0] = x
        for t in range(nsteps):
            x = self.step(x,dt)
            traj[t+1] = x
            
        return traj.reshape((1,nsteps+1))
    
class DoubleWellX:
    
    def __init__(self,beta,gamma):
        self.beta = beta
        self.gamma = gamma
        
    def _potential(self,x):
        return 3*(x[0,:]**4) - 5*(x[0,:]**2) + 1.5*x[0,:] + 3*x[1,:]**2
    
    def _gradient(self,x):
        dx = 12*(x[0,:]**3)-10*x[0,:]+1.5
        dy = 6*x[1,:]
        return np.vstack([dx,dy])
    
    def step(self,x,dt):
        ''' Evolve process by one timestep
        

        Parameters
        ----------
        x : array, shape = (2,), current position of process.
        dt : float, length of timestep in seconds.

        Returns
        -------
        x_next : float, next position of process.
        '''
        g = (1/self.gamma)
        x_next = x - g*np.array([12*(x[0]**3)-10*x[0]+1.5,6*x[1]])*dt + np.sqrt((2*g*dt)/self.beta)*np.random.randn(2)
        return x_next
    
    def trajectory(self,x0,dt,nsteps):
        '''
        

        Parameters
        ----------
        x0 : float, initial position.
        dt : float, length of timestep in seconds.
        nsteps : int, number of steps for trajectory.

        Returns
        -------
        traj : array, shape = (1,n_steps+1).

        '''
        traj = np.zeros((nsteps+1,2))
        x = x0
        traj[0] = x
        for t in range(nsteps):
            x = self.step(x,dt)
            traj[t+1] = x
            
        return traj
    
class WellSystem:
    
    def __init__(self,dim,N,a,b,c,d0,tau):
        self.dim = dim
        self.N = N
        self.a = a
        self.b = b
        self.c = c
        self.d0 = d0
        self.tau = tau
        
    def __repr__(self):
        return repr('Well System with' + self.N + ' atoms.')
    
    def _r(self,x):
        return cdist(x,x)
    
    def _potential(self,x):
        d = self._r(x)
        u = d - self.d0
        terms = 1/(2*self.tau)*(self.a*u + self.b*(u**2) + self.c*(u**4))
        np.fill_diagonal(terms,0)
        return terms.sum()
    
    def _gradient(self,x):
        d = self._r(x)
        u = d - self.d0
        
        dV = np.zeros((self.N,self.dim))
        for i in range(self.N):
            grad = 0
            for j in range(self.N):
                dd = u[i,j]
                dx = (x[i]-x[j])/d[i,j] if d[i,j] > 0 else 0
                grad += (1/(2*self.tau))*(self.a+(2*self.b*dd)+4*self.c*(dd**3))*dx
            dV[i,:] = grad
            
        return dV
    
    
