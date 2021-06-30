import mdtraj
import numpy as np
import scipy.linalg as scl

class Molecule:

    def __init__(self, N, beta, kb=1.0, rb=1.0, ka=1.0, ra=np.pi/2, kd=None):
        """
        Enable simulation of overdamped Langevin dynamics for an over-
        simplified "molecule":

        Parameters:
        -----------
        N, int: number of atoms
        beta, float: inverse temperature.
        kb, float: bond constant governing harmonic potentials along each
                    bond length between two adjacent atoms.
        rb, float: bond rest position along each bond length between two adjacent atoms.
        ka, float: angle constant governing harmonic potentials along each
                    bond angle between three adjacent atoms.
        ra, float: angle rest position along each bond angle between three adjacent atoms.
        kd, ndarray (N-3, 2) or None: apply dihedral potential for each dihedral
                    angle between four adjacent atoms. For each of the N-3 dihedrals,
                    it contains a pre-factor and the number of minima

        """
        # Make parameters known:
        self.N = N
        self.beta = beta
        self.kb = kb
        self.rb = rb
        self.ka = ka
        self.ra = ra
        self.kd = kd


    def simulate(self, m, dt, x0, stride=1):
        """
        Simulate the system for m discrete steps at integration time step dt, starting from x0.

        Parameters:
        -----------
        m, int: Number of discrete time steps.
        dt, float: Discrete integration time step.
        x0, ndarray(N, 3): initial positions.
        stride, int: Downsampling factor (optional)

        Returns:
        --------
        traj, ndarray(floor(m/stride) + 1, N, 3): simulated trajectory.
        """
        # Allocate trajectory:
        meff = int(np.floor(m/stride)) + 1
        X = np.zeros((meff, self.N, 3))
        # Initialize:
        x = x0
        X[0, :, :] = x0
        ll = 1
        # Run Euler Scheme:
        for kk in range(1, m + 1):
            # Evaluate gradient:
            g = self._gradient(x)
            # Compute next position:
            x = x - dt * g + np.sqrt(2 * dt / self.beta) * np.random.randn(self.N, 3)
            # Store:
            if np.remainder(kk, stride) == 0:
                X[ll, :, :] = x
                ll += 1
                
            if (kk+1)%100000 == 0:
                print('Reached step ' +str(kk+1))
        return X


    def _potential(self, x):
        """ Evaluate full potential.

        x, ndarray(N, 3): position vector.
        """
        # Evaluate all displacements between neighboring atoms:
        disp = self._displacements(x)
        V = 0.0
        # Evaluate all harmonic potentials:
        for ii in range(self.N - 1):
            V += self._harmonic_bond(disp[ii, :], self.kb, self.rb)
        # Evaluate all bond angle potentials:
        for ii in range(self.N - 2):
            V += self._harmonic_angle(disp[ii, :], disp[ii+1, :], self.ka, self.ra)
        # Evaluate all dihedral potentials:
        for ii in range(self.N -3):
            V += self._dihedral_potential(disp[ii, :], disp[ii+1, :], disp[ii+2, :],
                                          self.kd[ii, 0], self.kd[ii, 1])
        return V


    def _gradient(self, x):
        """ Evaluate full gradient:

        x, ndarray(N, 3): position vector.
        """
        # Evaluate all displacements between neighboring atoms:
        disp = self._displacements(x)
        g = np.zeros((self.N, 3))
        # Evaluate all gradients due to harmonic potentials:
        for ii in range(self.N - 1):
            g[ii:ii+2, :] += self._harmonic_bond_grad(disp[ii, :], self.kb, self.rb)
        # Evaluate all gradients due to harmonic angles:
        for ii in range(self.N - 2):
            g[ii:ii+3, :] += self._harmonic_angle_grad(disp[ii, :], disp[ii+1, :], self.ka, self.ra)
        # Evaluate all gradients due to dihedral angles:
            for ii in range(self.N - 3):
                g[ii:ii + 4, :] += self._dihedral_grad(disp[ii, :], disp[ii + 1, :], disp[ii + 2, :],
                                                       self.kd[ii, 0], self.kd[ii, 1])
        return g


    def _displacements(self, x):
        """
        Evaluate all displacement vectors between neighboring atoms.

        x, ndarray(N, 3): position vector.
        """
        disp = np.zeros((self.N - 1, 3))
        for ii in range(self.N - 1):
            disp[ii, :] = x[ii, :] - x[ii+1, :]

        return disp



    def _harmonic_bond(self, rij, k, r0):
        """ Evaluate harmonic potential at displacement vector rij,
            with constant k and rest position r0."""
        return 0.5 * k * (scl.norm(rij) - r0)**2

    def _harmonic_bond_grad(self, rij, k, r0):
        """ Evaluate gradient of harmonic bond potential at displacement rij,
            with constant k and rest position r0."""
        g = np.zeros((2, 3))
        r = scl.norm(rij)
        g[0, :] = rij / r
        g[1, :] = -rij / r
        return k * (r - r0) * g

    def _harmonic_angle(self, rij, rjk, k, theta0):
        """ Evaluate harmonic angle potential at displacement vectors rij, rjk
            with constant k and rest position theta0."""
        c_theta = np.dot(rij, -rjk) / (scl.norm(rij) * scl.norm(rjk))
        return 0.5 * k * (np.arccos(c_theta) - theta0) ** 2

    def _harmonic_angle_grad(self, rij, rjk, k, theta0):
        """ Evaluate gradient of harmonic bond potential at displacement rij,
            with constant k and rest position r0."""
        # Compute bond angle:
        c_theta = np.dot(rij, -rjk) / (scl.norm(rij) * scl.norm(rjk))
        theta = np.arccos(c_theta)
        # Compute gradient:
        r1 = scl.norm(rij)
        r2 = scl.norm(rjk)
        g = np.zeros((3, 3))
        g[0, :] = (-r1**2 * rjk - np.dot(rij, rjk)*rij) / (r1**3 * r2)
        g[2, :] = (r2**2 * rij + np.dot(rij, rjk)*rjk) / (r1 * r2**3)
        g[1, :] = -(g[0, :] + g[2, :])
        return -(k * (theta - theta0)) * (1.0/np.sqrt(1 - c_theta**2)) * g

    def _dihedral_potential(self, rij, rjk, rkl, k, n):
        """ Evaluate dihedral potential at displacements rij, rjk, rkl,
            with constant k and number of minima n. """
        r2 = scl.norm(rjk)
        phi = np.arctan2(np.dot(rjk, np.cross(np.cross(-rkl, rjk), np.cross(rij, rjk))),
                         r2 * np.dot(np.cross(rij, rjk), np.cross(-rkl, rjk)))
        return k * (1.0 - np.cos(n * phi))

    def _dihedral_grad(self, rij, rjk, rkl, k, n):
        """ Evaluate dihedral gradient in one-dimension at position phi,
            with constant k and number of minima n. """
        m1 = np.cross(rij, -rjk)
        m2 = np.cross(-rjk, rkl)
        m1n = scl.norm(m1)
        m2n = scl.norm(m2)
        r2 = scl.norm(rjk)
        phi = np.arctan2(np.dot(rjk, np.cross(np.cross(-rkl, rjk), np.cross(rij, rjk))),
                         r2 * np.dot(np.cross(rij, rjk), np.cross(-rkl, rjk)))

        g = np.zeros((4, 3))
        g[0, :] = (r2 / (m1n**2)) * m1
        g[3, :] = -(r2 / (m2n ** 2)) * m2
        g[1, :] = (np.dot(rij, -rjk)/(r2**2) - 1) * g[0, :] - (np.dot(-rjk, rkl)/(r2**2)) * g[3, :]
        g[2, :] = (np.dot(-rjk, rkl)/(r2**2) - 1) * g[3, :] - (np.dot(rij, -rjk)/r2**2) * g[0, :]
        return (k * n * np.sin(n * phi)) * g

    def _internal_coordinates(self, x):
        """ Compute all internal coordinates (bond lengths, bond angles, dihedrals
            for a given configuration x."""
        # Displacement vector:
        disp = self._displacements(x)
        # Prepare output:
        ic = np.zeros(3*self.N - 6)
        qq = 0
        # Evaluate all bond lenghts:
        for ii in range(self.N - 1):
            ic[qq] = scl.norm(disp[ii, :])
            qq += 1
        # Evaluate all bond angles:
        for ii in range(self.N - 2):
            c_theta = np.dot(disp[ii, :], -disp[ii+1, :]) / \
                      (scl.norm(disp[ii, :]) * scl.norm(disp[ii+1, :]))
            ic[qq] = np.arccos(c_theta)
            qq += 1
        # Evaluate all dihedral angles:
        for ii in range(self.N - 3):
            r2 = scl.norm(disp[ii, :]+1)
            phi = np.arctan2(np.dot(disp[ii+1, :], np.cross(np.cross(-disp[ii+2, :], disp[ii+1, :]),
                                                            np.cross(disp[ii, :], disp[ii+1, :]))),
                             r2 * np.dot(np.cross(disp[ii, :], disp[ii+1, :]), np.cross(-disp[ii+2, :],
                                                                                        disp[ii+1, :])))
            ic[qq] = phi
            qq += 1
        return ic

    def _create_top(self):
        # Initialize topology object:
        top = mdtraj.Topology()
        carbon = mdtraj.element.carbon
        ch1 = top.add_chain()
        # Add a residue:
        res1 = top.add_residue("RES1", ch1)
        # Add all atoms:
        atoms = []
        for ii in range(self.N):
            at_ii = top.add_atom(name="C%d" % ii, element=carbon, residue=res1)
            atoms.append(at_ii)
        # Add bonds:
        for ii in range(self.N - 1):
            top.add_bond(atoms[ii], atoms[ii + 1])
        return top
    
""" This function will create a topology object for visualization:"""
def create_top(N):
    # Initialize topology object:
    top = mdtraj.Topology()
    carbon = mdtraj.element.carbon
    ch1 = top.add_chain()
    # Add a residue:
    res1 = top.add_residue("RES1", ch1)
    # Add all atoms:
    atoms = []
    for ii in range(N):
        at_ii = top.add_atom(name="C%d"%ii, element=carbon, residue=res1)
        atoms.append(at_ii)
    # Add bonds:
    for ii in range(N-1):
        top.add_bond(atoms[ii], atoms[ii+1])
    return top



