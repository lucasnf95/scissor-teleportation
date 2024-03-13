
import numpy as np
import scipy as sp
import scipy.special as sps
import matplotlib.pyplot as plt

def mode_function(kappa, gamma, t0, t):
    return kappa*np.exp(-gamma*abs(t-t0)) - gamma*np.exp(-kappa*abs(t-t0))

def factorial(n):
    fact = 1
    if n == 0:
        return fact
    else:
        for i in range(n):
            fact *= i+1
    return fact

def a_operator(n_dim):
    a_operator = np.zeros((n_dim+1, n_dim+1))
    for j in range(n_dim):
        a_operator[j, j+1] = np.sqrt(j+1)
    return a_operator
        
def dagger(M):
    if type(M) == np.ndarray:
        dag = np.asarray(np.matrix(M).H)
    elif type(M) == np.matrix:
        dag = M.H
    else:
        print('Object is not a matrix or array')
    return dag

def wigner_function(density_matrix, n_dim):
    '''
    Compute Wigner function for specified dimension based on the density matrix given.
    '''
    q, p = [np.linspace(-10, 10, 200) for j in range(2)]
    P, Q = np.meshgrid(np.atleast_1d(p), np.atleast_1d(q))
    W = np.zeros((len(p), len(q)), dtype=complex)
    W_nm = np.zeros((len(p), len(q), n_dim+1, n_dim+1), dtype=complex) #expansion coefficients of the Wigner function
    X = 2*(Q**2 + P**2)
    #Compute the lower triangle of the matrix W_nm
    for n in np.arange(n_dim+1):
        for m in np.arange(n+1):
            dif = float(n-m)
            W_nm[:, :, n, m] = (1/np.pi)*np.exp(-0.5*X)*(-1)**m \
                               *(Q-1j*P)**dif*np.sqrt(2**dif*sp.special.gamma(m+1)/sp.special.gamma(n+1))\
                               *sp.special.assoc_laguerre(x=X, n=m, k=dif)     
    #Compute the upper triangle without the diagonal
    for m in np.arange(n_dim+1):
        for n in np.arange(m):
            W_nm[:, :, n, m] = np.conj(W_nm[:, :, m, n])

    #Compute the Wigner function
    for n in np.arange(n_dim+1):
        for m in np.arange(n_dim+1):
            W += W_nm[:, :, n, m]*density_matrix[n, m]
    #Get rid of nan values
    shape = W.shape
    W = W.flatten()
    W[np.where(np.isnan(W))] = 0
    W = W.reshape(shape)
    #Normalize in L1
    dq = q[1] - q[0]
    dp = p[1] - p[0]
    W *= np.pi/np.sum(np.sum(W) * dq*dp)
    
    return Q, P, W

def psi_coherent(alpha, theta, dimension):
    '''
    Compute 1 mode wavefunction for a coherent state
    Parameters:
        alpha: float
            Amplitude of the state
        theta: float
            Angle of the state in degrees
        dimension: int
            Fock state dimension of the computation
    Return
        psi: 1D array
            Wavefunction
    '''
    return [np.exp(-np.abs(alpha)**2/2)*(alpha*np.exp(1j*theta*np.pi/180))**n/np.sqrt(factorial(n)) for n in range(dimension+1)]

def rho_psi(psi):
    '''
    Compute 1 mode density matrix based on wavefunction
    Parameters:
        psi: 1D array
            Wavefunction
    Return:
        rho: 2D array
            Density matrix
    '''
    dimension = len(psi)
    rho = np.zeros((dimension,dimension), dtype=complex)
    for i in range(dimension):
        for j in range(dimension):
            rho[i, j] = psi[i]*np.conjugate(psi[j])
    return rho/np.trace(rho)

def iyad_homodyneFockPOVM(n_max, x, theta):
    """
    This function computes, the homodyne detection POVM in the Fock basis, up
    to order "N" (i.e, in a reduced-dimension Hilbert space). 
    
    For a quadrature value x, the (n, m) element of the POVM matrix is
    <X|m>*<X|n>. The expression of <X|k> depends on the associated LO phase
    "theta"
    
    n_max: positive integer
    x: scalar
    theta: scalar
    """
    #For each element of x, a POVM matrix is computed
    if n_max < 2:
        raise InvalidDimensionError('The Hilbert space dimension must be at least 2')
    POVM = np.zeros((n_max+1, ), dtype=complex)
    for n in np.arange(n_max):
        if n==0:
            POVM[n] = 1/(np.sqrt(np.sqrt(np.pi)))*np.exp(-0.5*x**2)
        elif n==1:
            POVM[n] = x*np.sqrt(2)*np.exp(-1j*theta) * POVM[0]
        else:
            POVM[n] = np.exp(-1j*theta)/np.sqrt(n)*(np.sqrt(2)*x*POVM[n-1] - np.exp(-1j*theta)*np.sqrt(n-1)*POVM[n-2])
    return np.tensordot(POVM, np.conj(POVM), axes=0)

def jonas_homodyneFockPOVM(n_max, x, theta):
    """
    This function computes the homodyne detection POVM in the Fock basis up
    to order "N" (i.e, in a reduced-dimension Hilbert space) using the Hermite polynomials.
    
    For a quadrature value x, the (n, m) element of the POVM matrix is
    <X|m>*<X|n>. The expression of <X|k> depends on the associated LO phase
    "theta"
    
    n_max: positive integer
    x: scalar
    theta: scalar
    """
    #For each element of x, a POVM matrix is computed
    if n_max < 2:
        raise InvalidDimensionError('The Hilbert space dimension must be at least 2')
    POVM = np.zeros((n_max+1, ), dtype=complex)
    for n in np.arange(n_max):
        POVM[n] = np.exp(-1j*n*theta)*sps.hermite(n)(x)*np.exp(-x*x/2.)/np.sqrt(np.sqrt(np.pi)*2**n*sps.gamma(n+1))
    return np.tensordot(POVM, np.conj(POVM), axes=0)
    

def isDensityMatrix(M, tolerance=1/100):
    """
    This function checks whether the matrix is a density matrix.
    M is a density matrix if and only if:

        - M is Hermitian
        - M is positive-semidefinite
        - M has trace 1
    """
    check_details = {}
    eigenvalues = np.linalg.eigvals(M)
    max_eigenval = np.max(np.abs(eigenvalues))
    check_details['is Hermitian'] = not np.any([not np.isclose(np.imag(e), 0) for e in eigenvalues])
    check_details['is positive-semidefinite'] = np.all([np.real(e)>=-tolerance*max_eigenval for e in eigenvalues])
    check_details['has trace 1'] = np.isclose(np.trace(M), 1)
    check = check_details['is Hermitian'] and check_details['is positive-semidefinite'] \
            and check_details['has trace 1']
    return check, check_details

def traceDistance(M_1, M_2):
    """
    Source: https://en.wikipedia.org/wiki/Trace_distance
    
    INPUTS
    ------------
        M_1 : 2-D array-like of complex
            Matrix 1
        M_2 : 2-D array-like of complex
            Matrix 2
    
    OUTPUTS
    -----------
    The trace distance
    """   
    X = M_1 - M_2
    return 0.5*np.trace(sp.linalg.sqrtm(np.transpose(np.conj(X)) @ X))


def quantumStateFidelity(rho_1, rho_2):
        """
        This function computes the fidelity of two quantum states, defined by
        the density matrices 'rho_1' and 'rho_2'.
        
        Source: https://en.wikipedia.org/wiki/Fidelity_of_quantum_states
        
        INPUTS
        ------------
            rho_1 : 2-D array-like of complex
                Density matrix 1
            rho_2 : 2-D array-like of complex
                Density matrix 2
        
        OUTPUTS
        -----------
        The fidelity (float in [0, 1])
        """                
        X = sp.linalg.sqrtm(rho_1) @ sp.linalg.sqrtm(rho_2)
        return (2*traceDistance(X, 0))**2

def hasConverged(rho_1, rho_2, convergence_parameter):
    convergence_metrics = np.real(quantumStateFidelity(rho_1, rho_2))
    converged = convergence_metrics > 1-convergence_parameter
    return converged, convergence_metrics

def calculate_projection_operators(data, dimension, tomography_phases = ['000', '030', '060', '090', '120', '150'], n_bins = 200):
    # Organize the quadrature measurements into bins
    # Compute homodyne projection operators in Fock basis

    # Maximum and minimum quadrature values for plotting
    quad_max = [np.max(data[phase]) for phase in tomography_phases]
    quad_max = np.max(quad_max)
    quad_min = [np.min(data[phase]) for phase in tomography_phases]
    quad_min = np.min(quad_min)

    x_bin_edges = np.linspace(quad_min, quad_max, n_bins+1)

    # Computing of projection operators and organization of quadrature measurements
    projection_operators = np.zeros((dimension+1, dimension+1, n_bins, len(tomography_phases)), dtype=complex)
    n_observations = np.zeros((n_bins, len(tomography_phases)))

    n_x = 1
    for p, phase in enumerate(tomography_phases):
        for j in range(n_bins):
            x = data[phase].flatten()
            data_bin = x[np.where(np.logical_and(x>x_bin_edges[j], x<x_bin_edges[j+1]))]
            x_bin_continuous = np.linspace(x_bin_edges[j], x_bin_edges[j+1], n_x)
            n_observations[j, p] = len(data_bin)
            projection_operator = np.zeros((dimension+1, dimension+1), dtype=complex)
            for k in range(n_x):
                jonas_projection_operator = jonas_homodyneFockPOVM(n_max = dimension, x = x_bin_continuous[k], theta = p)
                iyad_projection_operator = iyad_homodyneFockPOVM(n_max = dimension, x = x_bin_continuous[k], theta = p)
                projection_operator += iyad_projection_operator
            projection_operators[:,:,j,p] = (x_bin_edges[j+1] - x_bin_edges[j])*projection_operator
    return projection_operators, n_observations
            
def apply_maximum_likelihood(data, dimension, tomography_phases = ['000', '030', '060', '090', '120', '150'], convergence_parameter = 1e-8, n_bins = 200):
    # Apply the maximum likelihood method
    # Code is based on Iaji.Physics.Theory.QuantumMechanics.SimpleHarmonicOscillator.QuantumStateTomography

    converged = False
    projection_operators, n_observations = calculate_projection_operators(data, dimension, n_bins = n_bins)
    
    rho = (1/(dimension+1))*np.eye(dimension+1, dtype=complex) #estimated density operator
    past_log_likelihood = []
    run = 0
    while not converged:
        run += 1
        rho_last = rho
        R = np.zeros((dimension+1, dimension+1), dtype=complex)
        measurement_probabilities = np.zeros((n_bins, len(tomography_phases)))
        log_likelihood = 0
        for p in range(len(tomography_phases)):
            for j in range(n_bins):
                measurement_probabilities[j, p] = np.real(np.trace(projection_operators[:, :, j, p] @ rho))
        measurement_probabilities /= len(tomography_phases)
        R = np.tensordot(n_observations/measurement_probabilities, projection_operators, axes=([0, 1], [2, 3]))              
        R /= np.trace(R)
        log_likelihood = np.sum(n_observations*np.log(measurement_probabilities), (0, 1))
        R = (R+np.transpose(np.conj(R)))/2
        rho = R @ (rho @ R)
        rho /= np.trace(rho)
        #Check that it is a density matrix
        is_density_matrix = isDensityMatrix(rho)
        if not is_density_matrix[0]:
            raise NotADensityMatrixError(str(is_density_matrix[1]))
        past_log_likelihood.append(log_likelihood)
        #Check convergence
        converged, convergence_metrics = hasConverged(rho_1 = rho, rho_2 = rho_last, convergence_parameter=convergence_parameter)
    print('Converged')
    print('%d runs'%run)
    
    return rho

def calculate_output_state(data, input_rho, success_rate, dimension, plot = True, verbose = True):
    rho = apply_maximum_likelihood(data, dimension)
    
    # Calculate |alpha| and theta
    annihilation = a_operator(dimension)
    displacement = np.trace(annihilation @ rho)
    alpha = np.abs(displacement)
    angle = np.arctan(displacement.imag/displacement.real)*180/np.pi
    print('|alpha| = %.2f'%alpha)
    print('theta = %.2f°'%angle)
    
    if plot:
        # Plot density matrix
        plt.figure(figsize=(11, 8))
        #Define the maximum modulus of the density operator
        rho_max = np.max(np.abs(rho))
        #Plot
        plt.imshow(np.abs(rho), norm=matplotlib.colors.Normalize(vmin=-rho_max, vmax=rho_max))    
        plt.xlabel("n")
        plt.ylabel("m")
        plt.title("Density matrix")
        plt.pause(.05)
        # plt.colorbar()
        plt.show()
        print('Correlation term is', rho[0,1])

        # Plot photon number distribution
        figure = plt.figure(figsize=(11, 8))
        number = np.linspace(0, dimension, dimension+1)
        photon_number_distribution = [np.abs(rho[j, j]) for j in range(rho.shape[0])]
        #Plot
        plt.bar(number, photon_number_distribution)
        plt.xlabel("Photon number")
        plt.ylabel("Probability")
        plt.title("Output photon number distribution")
        plt.pause(.05)
        plt.show()
        print('Amount of single-photons:',photon_number_distribution[1])

        # Calculate Wigner function
        Q, P, W = wigner_function(rho, dimension)

        # Plot Wigner function
        W_max = np.max(np.abs(W))

        fig = plt.figure(figsize = [13,9])
        ax = plt.axes(projection='3d')
        W = np.real(W).astype(float)
        Q = Q.astype(float)
        P = P.astype(float)
        ax.plot_surface(Q, P, W, cmap='viridis', norm=matplotlib.colors.Normalize(vmin=-W_max, vmax=W_max))
        ax.set_title('Output state Wigner function')
        ax.set_xlabel('q')
        ax.set_ylabel('p')
        ax.set_zlabel('W(q,p)')
        fig.add_axes(ax)
        plt.show()

        # Countour plot of Wigner function
        fig = plt.figure(figsize=(9, 9) )
        ax = plt.axes()
        ax.contourf(Q, P, W, cmap='viridis', norm=matplotlib.colors.Normalize(vmin=-W_max, vmax=W_max))
        ax.set_title('Output state Wigner function')
        ax.set_xlabel('q')
        ax.set_ylabel('p')
        ax.grid()
        fig.add_axes(ax)
        #fig.colorbar(ax)
        plt.show()

    fidelity = quantumStateFidelity(rho, input_rho)
    if verbose:
        print("Output fidelity is %f" %fidelity)
        print("Success rate is %f" %np.mean(success_rate))
        print("The product of these values is %f" %(fidelity*np.mean(success_rate)))
        print("Purity of output state is %.2f" %np.trace(rho@rho))
    
    return rho, fidelity
