using AlgebraicOptimization
using Test
using Catlab
using ComponentArrays

# Test naturality of gradient descent: scalar variables
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

o1 = Euler(gradient_flow(p1), 0.1)
o2 = Euler(gradient_flow(p2), 0.1)
o3 = Euler(gradient_flow(p3), 0.1)

composite_of_optimizers = oapply(OpenDiscreteOpt(), d, [o1,o2,o3])

dc1 = Euler(optimizer_of_composite, 0.1)
dc2 = composite_of_optimizers

x0 = repeat([100.0], length(composite_prob.S))
tsteps = 1000
r1 = simulate(dc1, x0, tsteps)
r2 = simulate(dc2, x0, tsteps)

@test r1 ≈ r2


# Test ComponentArray version of input/output on scalar variables
# Note that all variables must be exposed to use this i/o system
d = @relation () begin
    f(a, b, c, d, e)
    g(f, g, a)
    h(b, a, f, h)
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

p1 = Open{PrimalObjective}(FinSet(5), PrimalObjective(FinSet(5),x->x'*P*x + a'*x))
p2 = Open{PrimalObjective}(FinSet(3), PrimalObjective(FinSet(3),x->x'*Q*x + b'*x))
p3 = Open{PrimalObjective}(FinSet(4), PrimalObjective(FinSet(4),x->x'*R*x + c'*x))

composite_prob = oapply(d, [p1,p2,p3])

optimizer_of_composite = gradient_flow(composite_prob)

o1 = Euler(gradient_flow(p1), 0.1)
o2 = Euler(gradient_flow(p2), 0.1)
o3 = Euler(gradient_flow(p3), 0.1)

composite_of_optimizers = oapply(OpenDiscreteOpt(), d, [o1,o2,o3])

dc1 = Euler(optimizer_of_composite, 0.1)
dc2 = composite_of_optimizers


x2 = ComponentArray(a=11, b=22, c=33, d=44, e=55, f=66, g=77, h=88, i=99)
tsteps = 1000
r1 = simulate(dc1, d, x2, tsteps)
r2 = simulate(dc2, d, x2, tsteps)

@test r1 ≈ r2




# Test naturality of gradient descent: vector variables
d = @relation (x,y,z) begin
    f(w,x)
    g(u,w,y)
    h(u,w,z)
end

f1 = x -> x[1][3] + x[4][1] - x[3][2] / x[2][1]
f2 = x -> 22 * x[2][2] / x[3][1]
f3 = x -> sum(sum(v) for v in x)

p1 = Open{PrimalObjective}(FinSet(5), PrimalObjective(FinSet(5), f1), FinFunction([2,4], 5))
p2 = Open{PrimalObjective}(FinSet(3), PrimalObjective(FinSet(3), f2), id(FinSet(3)))
p3 = Open{PrimalObjective}(FinSet(4), PrimalObjective(FinSet(4), f3), FinFunction([1,3,4]))

composite_prob = oapply(d, [p1,p2,p3])

optimizer_of_composite = gradient_flow(composite_prob)

o1 = Euler(gradient_flow(p1), 0.1)
o2 = Euler(gradient_flow(p2), 0.1)
o3 = Euler(gradient_flow(p3), 0.1)

composite_of_optimizers = oapply(OpenDiscreteOpt(), d, [o1,o2,o3])

dc1 = Euler(optimizer_of_composite, 0.1)
dc2 = composite_of_optimizers

x1::Vector{Vector{Float64}} = [[1, 1, 2], [2, 2], [3, 3], [4, 4], [1, 1], [2], [3], [4, 40, 40], [5]]   


tsteps = 1000
r1 = simulate(dc1, x1, tsteps)
r2 = simulate(dc2, x1, tsteps)

@test r1 ≈ r2


# Test ComponentArray version of input/output on vector variables
# Note that all variables must be exposed to use this i/o system
d = @relation () begin
    f(a, b, c, d, e)
    g(f, g, a)
    h(b, a, f, h)
end

f1 = x -> x[1][1] + x[4][1] - x[3][2] / x[2][1]
f2 = x -> 22 * x[2][2] / x[3][1]
f3 = x -> sum(sum(v) for v in x)


p1 = Open{PrimalObjective}(FinSet(5), PrimalObjective(FinSet(5), f1))
p2 = Open{PrimalObjective}(FinSet(3), PrimalObjective(FinSet(3), f2))
p3 = Open{PrimalObjective}(FinSet(4), PrimalObjective(FinSet(4), f3))


composite_prob = oapply(d, [p1,p2,p3])

optimizer_of_composite = gradient_flow(composite_prob)

o1 = Euler(gradient_flow(p1), 0.1)
o2 = Euler(gradient_flow(p2), 0.1)
o3 = Euler(gradient_flow(p3), 0.1)

composite_of_optimizers = oapply(OpenDiscreteOpt(), d, [o1,o2,o3])

dc1 = Euler(optimizer_of_composite, 0.1)
dc2 = composite_of_optimizers


x3 = ComponentArray(a=[2, 4, 6], b=[3, 5], c=[7, 9], d=[3, 4, 5, 6], e=[6, 6, 6], f=[8, 9], g=[10, 11, 12], h=[30, 33], i=[-2, 1.5])
tsteps = 1000
r1 = simulate(dc1, d, x3, tsteps)
r2 = simulate(dc2, d, x3, tsteps)

@test r1 ≈ r2