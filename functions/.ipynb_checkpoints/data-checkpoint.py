import numpy as np

# --------------------------
# Response function
# --------------------------
def swiss_roll_response(U, V):  
    """
    f(u,v) = 4*((u-7pi/2)/(3pi/2))^2 + pi/45*v
    """
    return 4 * ((U - 7*np.pi/2)/(3*np.pi/2))**2 + V*np.pi/45

# --------------------------
# Generate Swiss roll dataset
# --------------------------
def swiss_roll(n_data=200, noise_add=True, sigma_noise=0.1, f=None):
    X = np.zeros((3, n_data))
    U = np.random.uniform(low=2*np.pi/2, high=9*np.pi/2, size=n_data)
    V = np.random.uniform(low=0, high=15, size=n_data)
    X[0,:] = U * np.cos(U)
    X[1,:] = V
    X[2,:] = U * np.sin(U)
    Y = f(U,V)

    # Rescale X to [0,1]^3
    xr_boundary = [15, 15, 15]
    xl_boundary = [-15, 0, -15]
    for i in range(3):
        X[i,:] = (X[i,:] - xl_boundary[i]) / (xr_boundary[i]-xl_boundary[i])

    # Add noise to Y
    Y_noise = Y.copy()
    if noise_add:
        Y_noise += np.random.randn(n_data) * sigma_noise

    return X.T, Y, Y_noise

# --------------------------
# Generate Swiss roll with curve 
# --------------------------
def swiss_roll_curve(n_data=200, noise_add=True, sigma_noise=0.1, p=0.5, f=None):
    swiss_roll = np.zeros((3, n_data))
    n_swiss = np.random.binomial(n_data, p)
    U = np.random.uniform(low=2*np.pi/2, high=9*np.pi/2, size=n_swiss)
    V = np.random.uniform(low=0, high=15, size=n_swiss)

    # Straight Swiss roll part
    swiss_roll[0,:n_swiss] = U * np.cos(U)
    swiss_roll[1,:n_swiss] = V
    swiss_roll[2,:n_swiss] = U * np.sin(U)

    # Curved part
    t = np.random.uniform(low=-1, high=1, size=n_data - n_swiss)
    swiss_roll[0,n_swiss:] = 7*np.pi/2 * np.cos(np.pi * t) * np.cos(4 * np.pi * t)
    swiss_roll[1,n_swiss:] = 7*np.pi/2 + 7*np.pi/2 * np.cos(np.pi * t) * np.sin(4 * np.pi * t)
    swiss_roll[2,n_swiss:] = 7*np.pi/2 * np.sin(np.pi * t)

    # Output Y
    U_1 = np.sqrt(swiss_roll[0,n_swiss:]**2 + swiss_roll[2,n_swiss:]**2)
    V_1 = swiss_roll[1,n_swiss:]
    Y = np.zeros(n_data)
    Y[:n_swiss] = f(U,V)
    Y[n_swiss:] = f(U_1,V_1)
    Y_noise = Y.copy()

    # Rescale X
    xr_boundary = [15, 20, 15]
    xl_boundary = [-15, 0, -15]
    for i in range(3):
        swiss_roll[i,:] = (swiss_roll[i,:] - xl_boundary[i]) / (xr_boundary[i]-xl_boundary[i])

    # Add noise
    if noise_add:
        Y_noise += np.random.randn(n_data) * sigma_noise

    return swiss_roll.T, Y, Y_noise

# --------------------------
# Loader for simulated data
# --------------------------

def simulated_data_loader(*,train_size,n_data,sigmasq_0,
                            data_fn=None,f_fn=None):
    '''
    return
        X_train, X_test,Y_train, Y_de_train, Y_de_test
    '''
    if data_fn is None:
        raise ValueError("Please provide a data generation function using 'data_fn'.")
    
    X, Y_de, Y = data_fn(
        n_data=n_data,
        sigma_noise=np.sqrt(sigmasq_0),
        f=f_fn)

    idx = np.arange(n_data)
    np.random.shuffle(idx)

    train_idx = idx[:train_size]
    test_idx  = idx[train_size:]

    return(
        X[train_idx], X[test_idx],
        Y[train_idx], 
        Y_de[train_idx], Y_de[test_idx]
    )