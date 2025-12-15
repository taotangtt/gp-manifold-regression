import numpy as np
import scipy
import math
import time

def knn_prior_mcmc(train_size, replicas, n_data,sigmasq_0,
                      n_grid, left_end,right_end,grid_pos_tolerance,
                      a_0,b_0, gamma1, gamma2, 
                      burn_in, mcmc_time,
                      data_loader,data_loader_kwargs,
                      v_hat_method='arithmetic'):
          
    test_size = n_data - train_size
    k_data = 2 if train_size < 200 else math.ceil(np.log(train_size)**(2) * gamma2) #k of knn
    k_size = train_size if train_size < 200 else math.ceil(np.log(train_size)**3)  #|S|
    ls_grid_epsilon = np.linspace(left_end, right_end, n_grid)
    ls_w_pos_knn =  np.zeros(replicas)
    ls_w_pos_knn_test = np.zeros(replicas)
    log_grid_likelihoods = np.zeros(n_grid)
    log_grid_prior = np.zeros(n_grid)
    log_grid_pos = np.zeros(n_grid)
    Tn2_replicas = np.zeros(replicas)
    Tn_count_total = np.zeros(replicas)
    
    for replica in range(replicas):
        start = time.time()

        X_train, X_test,Y_train,Y_de_train, Y_de_test= data_loader(n_data=n_data,train_size=train_size,
                                                                   sigmasq_0=sigmasq_0,**data_loader_kwargs)
        
        
        #For original distance   
        dists_XX = scipy.spatial.distance.cdist(X_train, X_train)
        dists_pX = scipy.spatial.distance.cdist(X_test, X_train)
        dists_XX_sq = dists_XX**2
        dists_pX_sq = dists_pX**2
        
        k_nn_subtable = np.sort(dists_XX[:k_size, :], axis = 1)
        k_nn_distance_grid = (np.mean(k_nn_subtable[:,k_data]))**2 #Tn^2
        Tn2_replicas[replica] = k_nn_distance_grid
        
        # sigma proposal
        #Compute kernel for different bandwidth  
        for i, epsilon in enumerate(ls_grid_epsilon):
            if epsilon <= k_nn_distance_grid * gamma1:
                log_grid_pos[i] = -np.inf #p(t)=0 if t< Tn*gamma1
                continue
            k_XX_de =  np.exp(-0.5 * dists_XX_sq/epsilon)
            k_pX =  np.exp(-0.5 * dists_pX_sq/epsilon)
            k_XX = np.copy(k_XX_de)
            k_XX = k_XX + np.eye(train_size) *sigmasq_0

            L= np.linalg.cholesky(k_XX)
            hat_inverse = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            #hat_inverse = np.linalg.solve(k_XX, Y_train)
            
            log_grid_likelihoods[i] = -0.5 * np.matmul(Y_train, hat_inverse)
            log_grid_likelihoods[i] += -0.5 * 2.0 * np.sum(np.log(np.diag( L )))
    
            R_hat= (k_XX_de.sum(1)-1)/(k_XX_de.shape[1]-1)

            if v_hat_method == "arithmetic":
                v_hat_grid = np.mean(R_hat)
            elif v_hat_method == "harmonic":
                v_hat_grid = 1 / np.mean(1 / R_hat) 
            else:
                raise ValueError(f"Unknown v_hat_method: {v_hat_method}")
                        
            log_grid_prior[i] = - a_0 * np.log(epsilon) - b_0/v_hat_grid #our knn prior
    
            if epsilon > k_nn_distance_grid * gamma1:
                log_grid_pos[i] = log_grid_likelihoods[i] + log_grid_prior[i]
            else: 
                log_grid_pos[i] = -np.inf #p(t)=0 if t< Tn*gamma1
                
        weight_pos = np.exp(log_grid_pos - np.max(log_grid_pos))
        weight_pos = weight_pos/np.sum(weight_pos)
        pos_grid_maxp = ls_grid_epsilon[np.argmax(weight_pos)]
    
        ls_grid_select=ls_grid_epsilon[weight_pos>grid_pos_tolerance]
        sigma_proposal_knn =(max(ls_grid_select) - min(ls_grid_select))/25
    
        ls_epsilon = np.zeros(mcmc_time)
        log_pos_knn = np.zeros(mcmc_time)

        #set initial values
        ls_epsilon[0] = pos_grid_maxp
        epsilon = ls_epsilon[0]
        k_XX_de =  np.exp(-0.5 * dists_XX_sq/epsilon)
        k_pX =  np.exp(-0.5 * dists_pX_sq/epsilon)
        k_XX = np.copy(k_XX_de)
        k_XX = k_XX + np.eye(train_size) *sigmasq_0

        L= np.linalg.cholesky(k_XX)
        hat_inverse = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
        #hat_inverse = np.linalg.solve(k_XX, Y_train)
        
        log_likelihood = -0.5 * np.matmul(Y_train, hat_inverse)
        log_likelihood += -0.5 * 2.0 * np.sum(np.log(np.diag( L )))
        
        R_hat= (k_XX_de.sum(1)-1)/(k_XX_de.shape[1]-1)

        if v_hat_method == "arithmetic":
            v_hat = np.mean(R_hat)
        elif v_hat_method == "harmonic":
            v_hat = 1 / np.mean(1 / R_hat) 
        else:
            raise ValueError(f"Unknown v_hat_method: {v_hat_method}")
        
        log_pos_knn[0] = log_likelihood - a_0 * np.log(epsilon) - b_0/v_hat

        Y_pred_train = np.zeros((mcmc_time,train_size))
        Y_pred_test = np.zeros((mcmc_time,test_size))
        Tn_count = 0
        for i in range(1, mcmc_time):  
            #gaussian proposal
            epsilon = (np.random.randn(1)*sigma_proposal_knn + ls_epsilon[i-1])[0]
            if (epsilon < gamma1* k_nn_distance_grid) or (epsilon < left_end) or epsilon > right_end:
                ls_epsilon[i] = ls_epsilon[i-1]
                Y_pred_train[i,:] = Y_pred_train[i-1,:]
                Y_pred_test[i,:] = Y_pred_test[i-1,:]
                log_pos_knn[i] = log_pos_knn[i-1]
                if (epsilon < gamma1* k_nn_distance_grid):
                    Tn_count += 1
                continue
                
            k_XX_de =  np.exp(-0.5 * dists_XX_sq/epsilon)
            k_pX =  np.exp(-0.5 * dists_pX_sq/epsilon)
            k_XX = np.copy(k_XX_de)
            k_XX = k_XX + np.eye(train_size) *sigmasq_0

            L= np.linalg.cholesky(k_XX)
            hat_inverse = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            #hat_inverse = np.linalg.solve(k_XX, Y_train)
            
            log_likelihood = -0.5 * np.matmul(Y_train, hat_inverse)
            log_likelihood += -0.5 * 2.0 * np.sum(np.log(np.diag( L )))
            
            R_hat= (k_XX_de.sum(1)-1)/(k_XX_de.shape[1]-1)

            if v_hat_method == "arithmetic":
                v_hat = np.mean(R_hat)
            elif v_hat_method == "harmonic":
                v_hat = 1 / np.mean(1 / R_hat) 
            else:
                raise ValueError(f"Unknown v_hat_method: {v_hat_method}")
            
            log_pos_knn_can = log_likelihood - a_0 * np.log(epsilon) - b_0/v_hat
            log_pos_raio = min(0., log_pos_knn_can - log_pos_knn[i-1])
            accept_flag = np.log(np.random.uniform(0,1)) <= log_pos_raio
            if accept_flag:
                ls_epsilon[i] = epsilon
                Y_pred_train[i,:] = np.matmul(k_XX_de, hat_inverse)
                Y_pred_test[i,:] = np.matmul(k_pX, hat_inverse)
                log_pos_knn[i] = log_pos_knn_can
            else:
                ls_epsilon[i] = ls_epsilon[i-1]
                Y_pred_train[i,:] = Y_pred_train[i-1,:]
                Y_pred_test[i,:] = Y_pred_test[i-1,:]
                log_pos_knn[i] = log_pos_knn[i-1]
        Tn_count_total[replica] = Tn_count
        
        err_in = np.sqrt(np.mean((Y_de_train - np.mean(Y_pred_train[burn_in:], axis = 0))**2))
        err_te = np.sqrt(np.mean((Y_de_test - np.mean(Y_pred_test[burn_in:], axis = 0))**2))
        ls_w_pos_knn[replica] = err_in
        ls_w_pos_knn_test[replica] = err_te
        print(f"{replica} replica time: {(time.time() - start):.2f}, Tn_count {Tn_count}")
        print(f"proposal: {sigma_proposal_knn:.2e}, start_point: {pos_grid_maxp:.2f}, err_in: {err_in:.2f}, err_os: {err_te:.2f}")
    
    return ls_w_pos_knn, ls_w_pos_knn_test, Tn_count_total, log_grid_pos,ls_epsilon

def rescaled_gamma_prior_mcmc(train_size, replicas, n_data, sigmasq_0,
                           n_grid,left_end,right_end, grid_pos_tolerance,
                           a_0, b_0, d, 
                           burn_in, mcmc_time,
                           data_loader,data_loader_kwargs):
    
    test_size = n_data - train_size
    ls_grid_epsilon = np.linspace(left_end, right_end, n_grid)
    ls_w_pos =  np.zeros(replicas)
    ls_w_pos_test = np.zeros(replicas)
    log_grid_likelihoods = np.zeros(n_grid)
    log_grid_prior = np.zeros(n_grid)
    log_grid_pos = np.zeros(n_grid)
    log_grid_pos_knn = np.zeros(n_grid)
    
    for replica in range(replicas):
        start = time.time()

        X_train, X_test,Y_train,Y_de_train, Y_de_test= data_loader(n_data=n_data, train_size=train_size,
                                                                   sigmasq_0=sigmasq_0, **data_loader_kwargs)
        
        #For original distance   
        dists_XX = scipy.spatial.distance.cdist(X_train, X_train)
        dists_pX = scipy.spatial.distance.cdist(X_test, X_train)
        dists_XX_sq = dists_XX**2
        dists_pX_sq = dists_pX**2
    
        #Compute kernel for different bandwidth  
        for i, epsilon in enumerate(ls_grid_epsilon):
            k_XX_de =  np.exp(-0.5 * dists_XX_sq/epsilon)
            k_pX =  np.exp(-0.5 * dists_pX_sq/epsilon)
            
            k_XX = np.copy(k_XX_de)
            k_XX = k_XX + np.eye(train_size) *sigmasq_0

            # Cholesky decomposition (stable)    
            L= np.linalg.cholesky(k_XX)
            hat_inverse = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            #hat_inverse = np.linalg.solve(k_XX, Y_train)
            
            log_grid_likelihoods[i] = -0.5 * np.matmul(Y_train, hat_inverse)
            log_grid_likelihoods[i] += -0.5 * 2.0 * np.sum(np.log(np.diag( L  )))
            
            #log prior for varying d: (-a_0*d/2 -1)*log(epsilon) - b_0/epsilon**{d/2}
            log_grid_prior[i] = (-a_0*d/2 - 1)*np.log(epsilon)  - b_0/epsilon **(d/2)
            
            log_grid_pos[i] = log_grid_likelihoods[i] + log_grid_prior[i]
    
        weight_pos = np.exp(log_grid_pos - np.max(log_grid_pos))
        weight_pos = weight_pos/np.sum(weight_pos)
        pos_grid_maxp = ls_grid_epsilon[np.argmax(weight_pos)]
        
        ls_grid_select=ls_grid_epsilon[weight_pos>grid_pos_tolerance]
        sigma_proposal =(max(ls_grid_select) - min(ls_grid_select))/25
          
        ls_epsilon = np.zeros(mcmc_time)
        log_pos = np.zeros(mcmc_time)
        
        ls_epsilon[0] = pos_grid_maxp
        epsilon = ls_epsilon[0]
        k_XX_de =  np.exp(-0.5 * dists_XX_sq/epsilon)
        k_pX =  np.exp(-0.5 * dists_pX_sq/epsilon)

        
        k_XX = np.copy(k_XX_de)
        k_XX = k_XX + np.eye(train_size) *sigmasq_0

        L= np.linalg.cholesky(k_XX)
        hat_inverse = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
        #hat_inverse = np.linalg.solve(k_XX, Y_train)
        
        log_likelihood = -0.5 * np.matmul(Y_train, hat_inverse)
        log_likelihood += -0.5 * 2.0 * np.sum(np.log(np.diag( L )))
        
        log_pos[0] = log_likelihood + (-a_0*d/2 - 1)*np.log(epsilon)  - b_0/epsilon **(d/2)
        
        Y_pred_train = np.zeros((mcmc_time,train_size))
        Y_pred_test = np.zeros((mcmc_time,test_size))
        
        for i in range(1, mcmc_time):  
            epsilon = (np.random.randn(1)*sigma_proposal + ls_epsilon[i-1])[0]
            if (epsilon < left_end) or epsilon > right_end:
                ls_epsilon[i] = ls_epsilon[i-1]
                Y_pred_train[i,:] = Y_pred_train[i-1,:]
                Y_pred_test[i,:] = Y_pred_test[i-1,:]
                log_pos[i] = log_pos[i-1]
                continue
            k_XX_de =  np.exp(-0.5 * dists_XX_sq/epsilon)
            k_pX =  np.exp(-0.5 * dists_pX_sq/epsilon)
            
            k_XX = np.copy(k_XX_de)
            k_XX = k_XX + np.eye(train_size) *sigmasq_0

            L= np.linalg.cholesky(k_XX)
            hat_inverse = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))    
            #hat_inverse = np.linalg.solve(k_XX, Y_train)
            
            log_likelihood = -0.5 * np.matmul(Y_train, hat_inverse)
            log_likelihood += -0.5 * 2.0 * np.sum(np.log(np.diag( L )))

            log_pos_can = log_likelihood + (-a_0*d/2 - 1)*np.log(epsilon)  - b_0/epsilon **(d/2)
            log_pos_raio = min(0., log_pos_can - log_pos[i-1])
            accept_flag = np.log(np.random.uniform(0,1)) <= log_pos_raio
            if accept_flag:
                ls_epsilon[i] = epsilon      
                Y_pred_train[i,:] = np.matmul(k_XX_de, hat_inverse)
                Y_pred_test[i,:] = np.matmul(k_pX, hat_inverse)
                log_pos[i] = log_pos_can
            else:
                ls_epsilon[i] = ls_epsilon[i-1]
                Y_pred_train[i,:] = Y_pred_train[i-1,:]
                Y_pred_test[i,:] = Y_pred_test[i-1,:]
                log_pos[i] = log_pos[i-1]
        err_in = np.sqrt(np.mean((Y_de_train - np.mean(Y_pred_train[burn_in:], axis = 0))**2))
        err_te = np.sqrt(np.mean((Y_de_test - np.mean(Y_pred_test[burn_in:], axis = 0))**2))
        if np.isnan(err_te):
            break
        ls_w_pos[replica] = err_in
        ls_w_pos_test[replica] = err_te
        print(f"{replica} replica time: {(time.time() - start):.2f}")
        print(f"proposal: {(sigma_proposal):.2e}, start_point: {pos_grid_maxp:.2f}, err_in: {err_in:.2f}, err_os: {err_te:.2f}")
        
    return ls_w_pos,ls_w_pos_test, log_grid_pos,ls_epsilon

def kernel_ridge_cv(train_size, replicas,val_percent,n_data,sigmasq_0,
                           n_grid, left_end,right_end,
                           data_loader,data_loader_kwargs):
          
    test_size = n_data - train_size
    ls_single = np.zeros(replicas)
    ls_insample = np.zeros((replicas,n_grid))
    ls_outsample = np.zeros((replicas,n_grid))
    ls_grid_epsilon = np.linspace(left_end, right_end, n_grid)
    ls_val = np.zeros((replicas,n_grid))
    for replica in range(replicas):
        start = time.time()

        X_train, X_test,Y_train,Y_de_train, Y_de_test= data_loader(n_data=n_data,train_size=train_size,
                                                                   sigmasq_0=sigmasq_0,**data_loader_kwargs)
        
        #For original distance   
        dists_XX = scipy.spatial.distance.cdist(X_train, X_train)
        dists_pX = scipy.spatial.distance.cdist(X_test, X_train)
        dists_XX_sq = dists_XX**2
        dists_pX_sq = dists_pX**2
        val_size = int(np.ceil(train_size *val_percent))
       
        dists_XX_sq_kr,dists_vX_sq_kr, dists_pX_sq_kr = dists_XX_sq[val_size:,val_size:],dists_XX_sq[:val_size,val_size:],dists_pX_sq[:,val_size:]
        Y_de_train_kr,Y_de_val_kr = Y_de_train[val_size:],Y_de_train[:val_size]
        Y_train_kr,Y_val_kr = Y_train[val_size:], Y_train[:val_size]
        
        # single_means 
        ls_single[replica] = np.mean(np.power(Y_de_train - Y_train,2))
        
        #kernel ridge
        Y_pred_train = np.zeros((n_grid,train_size-val_size))
        Y_pred_test = np.zeros((n_grid,test_size))
        Y_pred_val = np.zeros((n_grid,val_size))
        for i, epsilon in enumerate(ls_grid_epsilon):
            k_XX_de =  np.exp(-0.5 * dists_XX_sq_kr/epsilon)
            k_pX =  np.exp(-0.5 * dists_pX_sq_kr/epsilon)
            k_XX = np.copy(k_XX_de)
            k_XX = k_XX + np.eye(train_size-val_size) *sigmasq_0
            k_vX = np.exp(-0.5 * dists_vX_sq_kr/epsilon)

            L= np.linalg.cholesky(k_XX)
            hat_inverse = np.linalg.solve(L.T, np.linalg.solve(L, Y_train_kr))
            #hat_inverse = np.linalg.solve(k_XX, Y_train_kr)
            
            Y_pred_train[i,:] = np.matmul(k_XX_de, hat_inverse)
            Y_pred_test[i,:] = np.matmul(k_pX, hat_inverse)
            Y_pred_val[i,:] = np.matmul(k_vX, hat_inverse)
            ls_insample[replica, i] = np.mean(np.power(Y_de_train_kr - Y_pred_train[i,:],2))
            ls_outsample[replica, i] = np.mean(np.power(Y_de_test - Y_pred_test[i,:],2))
            ls_val[replica, i] = np.mean(np.power(Y_de_val_kr - Y_pred_val[i,:],2))
    
        print(f"{replica} replica time: {(time.time() - start):.2f}")

    ls_single =np.sqrt(np.array(ls_single))
    ls_kernel_ridge =np.sqrt(np.array([ls_insample[enum, item] for enum, item in enumerate(np.argsort(ls_val)[:,0])]))
    ls_kernel_ridge_test = np.sqrt(np.array([ls_outsample[enum, item] for enum, item in enumerate(np.argsort(ls_val)[:,0])]))
    return ls_kernel_ridge, ls_kernel_ridge_test, ls_single
