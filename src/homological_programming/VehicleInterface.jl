module VehicleInterface

using LinearAlgebra
using ForwardDiff

export AbstractVehicleModel
export state_dim, control_dim, equilibrium, position_indices
export dynamics, linearize, compute_control
export LinearizedModel

"""
    AbstractVehicleModel

Interface for vehicle models. Implement for each concrete type:

    dynamics(model, x, u)   -> ẋ
    state_dim(model)        -> Int
    control_dim(model)      -> Int
    equilibrium(model)      -> (x_eq, u_eq)
    position_indices(model) -> AbstractVector{Int}

`linearize` and `LinearizedModel` are provided automatically via ForwardDiff.
"""
abstract type AbstractVehicleModel end

"""
    dynamics(model, x, u) -> ẋ

Continuous-time dynamics. Must be implemented for each concrete vehicle type.
"""
function dynamics(model::AbstractVehicleModel, x::AbstractVector, u::AbstractVector)
    error("dynamics not implemented for $(typeof(model))")
end

function state_dim(model::AbstractVehicleModel)
    error("state_dim not implemented for $(typeof(model))")
end

function control_dim(model::AbstractVehicleModel)
    error("control_dim not implemented for $(typeof(model))")
end

"""
    equilibrium(model) -> (x_eq, u_eq)

Nominal equilibrium state and input used for linearization and control.
"""
function equilibrium(model::AbstractVehicleModel)
    error("equilibrium not implemented for $(typeof(model))")
end

"""
    position_indices(model) -> AbstractVector{Int}

Indices into the state vector corresponding to Cartesian position.
Used by the sheaf coordination layer to extract positions without knowing
the vehicle's internal state layout. Default: 1:3.
"""
position_indices(model::AbstractVehicleModel) = 1:3

"""
    linearize(model, x0, u0) -> (A, B)
    linearize(model)         -> (A, B)

Linearize dynamics at (x0, u0) via ForwardDiff. A = ∂f/∂x, B = ∂f/∂u.
Calling without arguments linearizes at the model's equilibrium.
"""
function linearize(model::AbstractVehicleModel, x0::AbstractVector, u0::AbstractVector)
    A = ForwardDiff.jacobian(x -> dynamics(model, x, u0), x0)
    B = ForwardDiff.jacobian(u -> dynamics(model, x0, u), u0)
    return A, B
end

function linearize(model::AbstractVehicleModel)
    x0, u0 = equilibrium(model)
    return linearize(model, x0, u0)
end

"""
    compute_control(ctrl, x, x_ref) -> u

Evaluate the controller at state x tracking reference x_ref.
Defined here so all controller modules extend a single canonical function.
"""
function compute_control(ctrl, x::AbstractVector, x_ref::AbstractVector)
    error("compute_control not implemented for $(typeof(ctrl))")
end

compute_control(ctrl::Function, x::AbstractVector, x_ref::AbstractVector) = ctrl(x, x_ref)

"""
    LinearizedModel <: AbstractVehicleModel

Wraps any `AbstractVehicleModel` with its linearization computed once at
construction via ForwardDiff. All simulation uses the linear dynamics:

    ẋ = A·(x - x_eq) + B·(u - u_eq)

Construct with `LinearizedModel(model)`. Controller constructors should still
receive the original nonlinear model so they can linearize internally as needed.

Ref: Khalil, "Nonlinear Systems", 3rd ed., §4.3 (linearization).
"""
struct LinearizedModel <: AbstractVehicleModel
    A::Matrix{Float64}
    B::Matrix{Float64}
    x_eq::Vector{Float64}
    u_eq::Vector{Float64}
    _state_dim::Int
    _control_dim::Int
    _position_indices::UnitRange{Int}
end

function LinearizedModel(model::AbstractVehicleModel)
    x_eq, u_eq = equilibrium(model)
    A, B = linearize(model, x_eq, u_eq)
    return LinearizedModel(A, B, x_eq, u_eq,
        state_dim(model), control_dim(model), position_indices(model))
end

state_dim(m::LinearizedModel) = m._state_dim
control_dim(m::LinearizedModel) = m._control_dim
position_indices(m::LinearizedModel) = m._position_indices
equilibrium(m::LinearizedModel) = (m.x_eq, m.u_eq)

function dynamics(m::LinearizedModel, x::AbstractVector, u::AbstractVector)
    return m.A * (x .- m.x_eq) .+ m.B * (u .- m.u_eq)
end

end
