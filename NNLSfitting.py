function [s, s2, mu] = NNLSfitting(DBasis, signal)

    # NNLS fitting w\ Reg (find unknown signal amplitude of components (signal)) minimises norm(A*s-signal) for reference
    s = lsqnonneg(DBasis,signal) 

    # Regularization fitting NNLS
    [s2, mu, resid] = CVNNLS(DBasis,signal) # CVNNLS from Bjarnason
    #fprintf('� = #f \n', mu) # larger mu = more satisfaction of constraints at expanse of increasing misfit (Witthal1989)
    
end