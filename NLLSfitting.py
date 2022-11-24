function [d, f, resnorm] = NLLSfitting(b, signal, dIn, fIn, Dmin, Dmax)
# NLLSfitting(inputSimu, dIn, fIn) =  a priori information dNNLS and fNNLS in x0
# NLLSfitting(inputSimu) = no a priori information, using standard start value

    if nargin < 3
        x0 = [1.35*1e-3, 4*1e-3, 155*1e-3, 52.5, 40]'  # default start values for triExp [Periquito2021]
    else
        input = [dIn fIn]
        x0 = input(1:5)
    end
    
    np = nnz(x0(1:3))                                  # number of found compartments by NNLS   
    options.Algorithm = 'levenberg-marquardt' 
    options.Display = 'off'

    lb = [repelem(Dmin,np) repelem(0,np-1)]            # set bound constraints based on NNLS d range
    ub = [repelem(Dmax,np) repelem(100,np-1)]
    scaling = 100/signal(1)                            # scale signal for NLLS to find reasonable volume fractions
    signal = signal.*scaling
        
    if     np == 3
        # Create tri-exponential signal function for fitting with d and f as fitting variable
        triExp = @(x) exp(-kron(b, abs(x(1))))*x(4) + exp(-kron(b, abs(x(2))))*x(5) + exp(-kron(b, abs(x(3))))*(100-(x(4)+x(5))) - signal
        [s(1:5), resnorm(:)] = lsqnonlin(triExp, x0, lb, ub, options) 
        s(6) =(100-(s(4)+s(5)))

    elseif np == 2
        # Create bi-exponential signal function
        biExp = @(x) exp(-kron(b, abs(x(1))))*x(4) + exp(-kron(b, abs(x(2))))*(100-x(4)) - signal
        [s(1:4), resnorm(:)] = lsqnonlin(biExp, x0(1:4),[],[],options) 
        s(5) =(100-s(4))
        s(6) = 0

    elseif np == 1
        # Create mono-exponential signal function
        monoExp = @(x) exp(-kron(b, abs(x(1)))) - signal
        [s(1), resnorm(:)] = lsqnonlin(monoExp, x0(1),[],[],options) 
        s(2:6) = zeros(1,5)
        s(4) = 1
    
    elseif np == 4 
        # Create 4-exponential signal function
        quadExp = @(x) exp(-kron(b, abs(x(1))))*x(5) + exp(-kron(b, abs(x(2))))*x(6) + exp(-kron(b, abs(x(3))))*x(7) + exp(-kron(b, abs(x(4))))*(100-(x(5)+x(6)+x(7))) - signal
        [s, resnorm(:)] = lsqnonlin(quadExp, x0, lb, ub, options) 
        s(8) =(100-(s(5)+s(6)+s(7)))
    end

    d = abs(s(1:np))
    f = s(np+1:end)
end