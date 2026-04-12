using Test
using AlgebraicOptimization
using LinearAlgebra
using ControlSystems

@testset "QuadrotorLQR" begin

    p = DEFAULT_PARAMS

    # ── LQR gain ─────────────────────────────────────────────────────────────
    @testset "lqr gain" begin
        # Double integrator — known solution
        A = [0.0 1.0; 0.0 0.0]
        B = reshape([0.0; 1.0], 2, 1)
        Q = Matrix(1.0 * I(2))
        R = reshape([1.0], 1, 1)

        K = lqr(A, B, Q, R)
        @test size(K) == (1, 2)
        @test all(real.(eigvals(A - B * K)) .< 0)
    end

    # ── LQRController construction ───────────────────────────────────────────
    @testset "LQRController" begin
        ctrl = LQRController(p)
        @test size(ctrl.K_pos) == (2, 4)
        @test size(ctrl.K_att) == (3, 6)
        @test size(ctrl.K_z)   == (1, 2)
    end

    # ── compute_control ──────────────────────────────────────────────────────
    @testset "compute_control" begin
        ctrl = LQRController(p)

        # At hover with zero error the virtual inputs should equal nominal
        x0, _ = equilibrium(QuadrotorModel())
        U = compute_control(ctrl, x0, x0)

        @test U[1] ≈ p.m * p.g  atol=1e-8    # total thrust = weight
        @test U[2] ≈ 0.0        atol=1e-8
        @test U[3] ≈ 0.0        atol=1e-8
        @test U[4] ≈ 0.0        atol=1e-8

        # Perturb altitude: controller should command more thrust
        x_high = copy(x0); x_high[3] = 1.0   # z error = +1 m (above ref)
        U_high = compute_control(ctrl, x_high, x0)
        @test U_high[1] < p.m * p.g           # reduce thrust to descend
    end

    # ── QuadrotorModel (VehicleInterface) ────────────────────────────────────
    @testset "QuadrotorModel" begin
        model = QuadrotorModel()

        @test state_dim(model)   == 12
        @test control_dim(model) == 4
        @test position_indices(model) == 1:3

        x0, u0 = equilibrium(model)
        @test length(x0) == 12
        @test all(x0 .== 0.0)
        @test u0[1] ≈ p.m * p.g  atol=1e-10
        @test all(u0[2:4] .== 0.0)

        # dynamics at equilibrium should return zero derivative
        ẋ = dynamics(model, x0, u0)
        @test norm(ẋ) ≈ 0.0  atol=1e-10

        # linearize via ForwardDiff should produce valid A, B matrices
        A, B = linearize(model)
        @test size(A) == (12, 12)
        @test size(B) == (12, 4)
    end

    # ── SheafControllerInterface ─────────────────────────────────────────────
    @testset "SheafControllerInterface" begin
        ctrl  = LQRController(p)
        iface = SheafControllerInterface(ctrl, QuadrotorModel(); planner_hz=10.0)

        x0, _ = equilibrium(QuadrotorModel())

        # Before any planner update x_ref = 0 (hover)
        U = step!(iface, x0)

        @test U[1] ≈ p.m * p.g  atol=1e-8

        # Planner updates reference (e.g. move 1 m in x)
        x_ref_new = copy(x0); x_ref_new[1] = 1.0
        update_reference!(iface, x_ref_new, 0.1)
        @test iface.x_ref[1] ≈ 1.0

        # Inner loop step with the new reference
        U = step!(iface, x0)
        @test length(U) == 4
    end

end
