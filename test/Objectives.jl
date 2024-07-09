using AlgebraicOptimization
using Test
using Catlab

# Test naturality of gradient descent
d = @relation (x,y,z) begin
    f(w,x)
    g(u,w,y)
    h(u,w,z)
end


P = [2.1154  -0.3038   0.368   -1.5728  -1.203
 -0.3038   1.5697   1.0226   0.159   -0.946
  0.368    1.0226   1.847   -0.4916  -1.2668
 -1.5728   0.159   -0.4916   2.2192   1.5315
 -1.203   -0.946   -1.2668   1.5315   1.9281]
Q = [0.2456   0.3564  -0.0088
  0.3564   0.5912  -0.0914
 -0.0088  -0.0914   0.8774]
R = [2.0546  -1.333   -0.5263   0.3189
 -1.333    1.0481  -0.0211   0.2462
 -0.5263  -0.0211   0.951   -0.7813
  0.3189   0.2462  -0.7813   1.5813]

a = [-0.26, 0.22, 0.09, 0.19, -0.96]
b = [-0.72, 0.12, 0.41]
c = [0.55, 0.51, 0.6, -0.61]

p1 = Open{PrimalObjective}(FinSet(5), PrimalObjective(FinSet(5),x->x'*P*x + a'*x), FinFunction([2,4], 5))
p2 = Open{PrimalObjective}(FinSet(3), PrimalObjective(FinSet(3),x->x'*Q*x + b'*x), id(FinSet(3)))
p3 = Open{PrimalObjective}(FinSet(4), PrimalObjective(FinSet(4),x->x'*R*x + c'*x), FinFunction([1,3,4]))

composite_prob = oapply(d, [p1,p2,p3])

optimizer_of_composite = gradient_flow(composite_prob)

o1 = gradient_flow(p1)
o2 = gradient_flow(p2)
o3 = gradient_flow(p3)

composite_of_optimizers = oapply(OpenContinuousOpt(), d, [o1,o2,o3])

dc1 = Euler(optimizer_of_composite, 0.1)
dc2 = Euler(composite_of_optimizers, 0.1)

x0 = repeat([100.0], length(composite_prob.S))
tsteps = 1000
r1 = simulate(dc1, x0, tsteps)
r2 = simulate(dc2, x0, tsteps)

@test r1 ≈ r2