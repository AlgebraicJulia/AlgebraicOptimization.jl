using AlgebraicOptimization.Objectives
using AlgebraicOptimization.Optimizers
using Test
using Catlab

# Test naturality of gradient descent
d = @relation (x,y,z) begin
    f(w,x)
    g(u,w,y)
    h(u,w,z)
end

P = rand(-1:0.01:1,5,5)
P = P'*P
Q = rand(-1:0.01:1,3,3)
Q = Q'*Q
R = rand(-1:0.01:1,4,4)
R = R'*R

a = rand(-1:0.01:1,5)
b = rand(-1:0.01:1,3)
c = rand(-1:0.01:1,4)

p1 = Open{PrimalObjective}(FinSet(5), x->x'*P*x + a'*x, FinFunction([2,4], 5))
p2 = Open{PrimalObjective}(FinSet(3), x->x'*Q*x + b'*x, id(FinSet(3)))
p3 = Open{PrimalObjective}(FinSet(4), x->x'*R*x + c'*x, FinFunction([1,3,4]))

composite_prob = oapply(d, [p1,p2,p3])

optimizer_of_composite = gradient_flow(composite_prob)

o1 = gradient_flow(p1)
o2 = gradient_flow(p2)
o3 = gradient_flow(p3)

composite_of_optimizers = oapply(OpenContinuousOpt(), d, [o1,o2,o3])



