using AlgebraicOptimization
using ForwardDiff
using Catlab

struct ContinuousSecondOrderSystem
    dim::FinSet
    impl::Function # R^dim × R^dim → R^dim × R^dim
end

struct DiscreteSecondOrderSystem
    dim::FinSet
    impl::Function
end

function eulers_method(s::ContinuousSecondOrderSystem, ss::Float64)::DiscreteSecondOrderSystem
    f = s.impl

    df = (xk, yk) -> begin
        delta_x, delta_y = f(xk, yk)
        return (xk + ss * delta_x, yk + ss * delta_y)
    end
    return DiscreteSecondOrderSystem(s.dim, df)
end

function trajectory(s::DiscreteSecondOrderSystem, x0, y0, niters)
    res_x = []
    res_y = []
    push!(res_x, x0)
    push!(res_y, y0)

    for i in 1:niters
        xup, yup = s.impl(res_x[end], res_y[end])
        push!(res_x, xup)
        push!(res_y, yup)
    end
    return res_x, res_y
end

function momentum(f::PrimalObjective)
    return ContinuousSecondOrderSystem(f.decision_space,
        (w, v) -> (-v, -v + ForwardDiff.gradient(f, w)))
end

A = rand(2, 2)
A = A' * A
b = rand(2)

B = rand(2, 2)
B = B' * B
c = rand(2)

f(x) = x' * A * x - b' * x
g(y) = y' * B * y - c' * y

h(z) = f(z[1:2]) + g(z[2:3])

f_obj = PrimalObjective(FinSet(2), f)
g_obj = PrimalObjective(FinSet(2), g)
h_obj = PrimalObjective(FinSet(3), h)

f_momentum = eulers_method(momentum(f_obj), 0.1)

f_w, f_v = trajectory(f_momentum, [100.0, 100.0], zeros(2), 200)

f_loss = f.(f_w)

plot(f_loss)

