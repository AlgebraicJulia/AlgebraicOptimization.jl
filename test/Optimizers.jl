using Revise
using AlgebraicOptimization
using Catlab
using Test

# Test naturality of Euler
d = @relation (x,y,z,u,w) begin    # Do I need to type these variables? Also, why does this low key take forever?
    f(w,x)
    g(u,w,y)
    h(u,w,z)
end

A = rand(-1:0.01:1,5,5)
B = rand(-1:0.01:1,3,3)
C = rand(-1:0.01:1,4,4)
fA = v -> [v[1] * v[1], v[2] + v[3], 4, 4, v[5]/v[4]]


o1 = Open{Optimizer}(FinSet(5), x->A*x, FinFunction([2,4], 5) )
o2 = Open{Optimizer}(FinSet(3), x->B*x, id(FinSet(3)))
o3 = Open{Optimizer}(FinSet(4), x->C*x, FinFunction([1,3,4]))

composite_opt = oapply(OpenContinuousOpt(), d, [o1,o2,o3])

discretization_of_composites = Euler(composite_opt, 0.01)

do1 = Euler(o1, 0.01)
do2 = Euler(o2, 0.01)
do3 = Euler(o3, 0.01)

composite_of_discretizations = oapply(OpenDiscreteOpt(), d, [do1,do2,do3])

x0 = repeat([1.0], length(composite_opt.S))
x2 = repeat([5.0], length(composite_opt.S))

tsteps = 100
r1 = simulate(discretization_of_composites, x0, tsteps)
r3 = simulate(discretization_of_composites, x2, tsteps)


r2 = simulate(composite_of_discretizations, x0, tsteps)

println(r3 ./ r1)

#println(r1)
#println(r2)

@test r1 ≈ r2




# Vector functions testing

mA = v -> v
mB = v -> [ [v[1][1] * v[3][1], 10, v[1][3] / v[2][1]], [v[2][1]], [4, 5] ]
mC = v -> v

# Why did we lose dimensions?
# Component arrays: typed by symbols  (go back and forth between a vector and a collection of vector)


o1 = Open{Optimizer}(FinSet(5), mA, FinFunction([2,4], 5))
o2 = Open{Optimizer}(FinSet(3), mB, id(FinSet(3)))
o3 = Open{Optimizer}(FinSet(4), mC, FinFunction([1,3,4]))


# Wrong domain sizes! Domain size of first function is # vars in f, codom size is # vars in the finset
# o1 = Open{Optimizer}(FinSet(5), mA, id(FinSet(5)))
# o2 = Open{Optimizer}(FinSet(3), mB, id(FinSet(3)))
# o3 = Open{Optimizer}(FinSet(4), mC, id(FinSet(4)))




composite_opt = oapply(OpenContinuousOpt(), d, [o1,o2,o3])

discretization_of_composites = Euler(composite_opt, 0.01)

do1 = Euler(o1, 0.01)
do2 = Euler(o2, 0.01)
do3 = Euler(o3, 0.01)

composite_of_discretizations = oapply(OpenDiscreteOpt(), d, [do1,do2,do3])

# x0 = repeat([1.0], length(composite_opt.S))
x1::Vector{Vector{Float64}} = [[1], [1], [1], [1], [1], [1, 1, 1], [1], [1, 1], [1], [1], [1], [1]]    # Checker would be helpful
tsteps = 100
r1 = simulate(discretization_of_composites, x1, tsteps)
r2 = simulate(composite_of_discretizations, x1, tsteps)
r2 - r1

#println(r1)
#println(r2)

@test r1 ≈ r2


# Various unit tests


f = FinFunction([1, 2, 2, 3])
domType = FinFunction([3, 2, 2, 1])
codomType = FinFunction([3, 2, 1])

domOnes = FinFunction([1, 1, 1, 1])
codomOnes = FinFunction([1, 1, 1])

typed_pullback_matrix(f, domType, codomType)

typed_pullback_matrix(f)

typed_pullback_matrix(f, domOnes, codomOnes)


f = FinFunction([1, 2, 2, 3])
v = [[33, 34, 35], [44, 45], [55, 56], [100]]
test_pullback_function(f, v)

test_pushforward_function(f, v)


wTest::Vector{Float64} = [20, 22, 33, 11]
test_pushforward_function(f, wTest)


parsed = @relation (x,y,z) where (x::1, y::2, z::3, w::4) begin    # Why is it still using the old modules??
  R(x,w)
  S(y,w)
  T(z,w)
end

draw(parsed)
draw_types(parsed)

draw(d)