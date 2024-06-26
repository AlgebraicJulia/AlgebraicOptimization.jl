using AlgebraicOptimization
using Catlab
using Test

# Test naturality of Euler
d = @relation (x,y,z) begin
    f(w,x)
    g(u,w,y)
    h(u,w,z)
end

A = rand(-1:0.01:1,5,5)
B = rand(-1:0.01:1,3,3)
C = rand(-1:0.01:1,4,4)

o1 = Open{Optimizer}(FinSet(5), x->A*x, FinFunction([2,4], 5))
o2 = Open{Optimizer}(FinSet(3), x->B*x, id(FinSet(3)))
o3 = Open{Optimizer}(FinSet(4), x->C*x, FinFunction([1,3,4]))

composite_opt = oapply(OpenContinuousOpt(), d, [o1,o2,o3])

discretization_of_composites = Euler(composite_opt, 0.01)

do1 = Euler(o1, 0.01)
do2 = Euler(o2, 0.01)
do3 = Euler(o3, 0.01)

composite_of_discretizations = oapply(OpenDiscreteOpt(), d, [do1,do2,do3])

x0 = repeat([1.0], length(composite_opt.S))
tsteps = 100
r1 = simulate(discretization_of_composites, x0, tsteps)
r2 = simulate(composite_of_discretizations, x0, tsteps)

#println(r1)
#println(r2)

@test r1 ≈ r2



