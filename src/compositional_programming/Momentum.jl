using AlgebraicOptimization
using ForwardDiff
using Catlab
using Plots

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

function adagrad(f::PrimalObjective)
    return ContinuousSecondOrderSystem(f.decision_space,
        (w, v) -> begin
            grad = ForwardDiff.gradient(f, w)
            return (-v, -v + grad / sqrt(sum(grad .^ 2) + 1e-8))
        end)


end

A = rand(2, 2)
A = A' * A
b = rand(2)

B = rand(2, 2)
B = B' * B
c = rand(2)

C = rand(2, 2)
C = C' * C
d = rand(2)

f(x) = x' * A * x - b' * x
g(y) = y' * B * y - c' * y
h(z) = z' * C * z - d' * z

h(z) = f(z[1:2]) + g(z[2:3])

f_obj = PrimalObjective(FinSet(2), f)
g_obj = PrimalObjective(FinSet(2), g)
h_obj = PrimalObjective(FinSet(3), h)

f_momentum_continuous = momentum(f_obj)
g_momentum_continuous = momentum(g_obj)

comp_momentum = ContinuousSecondOrderSystem(FinSet(3), (w, v) -> begin
    w1, w2 = w[1:2], w[2:3]
    v1, v2 = v[1:2], v[2:3]

    # Compute the momentum updates for f and g
    f_update_w, f_update_v = f_momentum_continuous.impl(w1, v1)
    g_update_w, g_update_v = g_momentum_continuous.impl(w2, v2)

    # Combine the momentum updates for f and g
    w_update = [f_update_w[1], f_update_w[2] + g_update_w[1], g_update_w[2]]
    v_update = [f_update_v[1], f_update_v[2] + g_update_v[1], g_update_v[2]]

    return (w_update, v_update)
end)

f_momentum = eulers_method(momentum(f_obj), 0.1)
g_momentum = eulers_method(momentum(g_obj), 0.1)
h_momentum = eulers_method(momentum(h_obj), 0.1)
comp_momentum = eulers_method(comp_momentum, 0.1)

h_w, h_v = trajectory(h_momentum, [100.0, 100.0, 100.0], zeros(3), 100)
comp_w, comp_v = trajectory(comp_momentum, [100.0, 100.0, 100.0], zeros(3), 100)

h_loss = h.(h_w)
comp_loss = h.(comp_w)

plot(h_loss, label="Momentum Composite")
plot!(comp_loss, label="Composite Momentum")

# Non-convex tests

ackley(u) = -20 * exp(-0.2 * sqrt(0.5 * (u[1]^2 + u[2]^2))) - exp(0.5 * (cos(2 * pi * u[1]) + cos(2 * pi * u[2]))) + exp(1) + 20

f_obj_ackley = PrimalObjective(FinSet(2), ackley)
g_obj_ackley = PrimalObjective(FinSet(2), ackley)
h_obj_ackley = PrimalObjective(FinSet(3), u -> ackley(u[1:2]) + ackley(u[2:3]))

f_momentum_ackley = momentum(f_obj_ackley)
g_momentum_ackley = momentum(g_obj_ackley)
h_momentum_ackley = eulers_method(momentum(h_obj_ackley), 0.1)
comp_momentum_ackley = ContinuousSecondOrderSystem(FinSet(3), (w, v) -> begin
    w1, w2 = w[1:2], w[2:3]
    v1, v2 = v[1:2], v[2:3]

    # Compute the momentum updates for f and g
    f_update_w, f_update_v = f_momentum_ackley.impl(w1, v1)
    g_update_w, g_update_v = g_momentum_ackley.impl(w2, v2)

    # Combine the momentum updates for f and g
    w_update = [f_update_w[1], f_update_w[2] + g_update_w[1], g_update_w[2]]
    v_update = [f_update_v[1], f_update_v[2] + g_update_v[1], g_update_v[2]]
    return (w_update, v_update)
end)

comp_momentum_ackley = eulers_method(comp_momentum_ackley, 0.1)


h_w, h_v = trajectory(h_momentum_ackley, [4.9, 4.9, 4.9], zeros(3), 100)
comp_w, comp_v = trajectory(comp_momentum_ackley, [4.9, 4.9, 4.9], zeros(3), 100)

h_loss = h_obj_ackley.(h_w)
comp_loss = h_obj_ackley.(comp_w)

plot(h_loss, label="Momentum Composite Ackley")
plot!(comp_loss, label="Composite Momentum Ackley")

#f_w, f_v = trajectory(f_momentum, [100.0, 100.0], zeros(2), 200)

#f_loss = f.(f_w)

#plot(f_loss)

