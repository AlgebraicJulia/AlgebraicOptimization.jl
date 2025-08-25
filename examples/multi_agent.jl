using AlgebraicOptimization
using JuMP

dt = 0.1
A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
B = [0 0; dt 0; 0 0; 0 dt]

R = I(2)

N = 10
control_bounds = [-2.0, 2.0]

struct MPCModel
    impl::JuMP.Model
end


