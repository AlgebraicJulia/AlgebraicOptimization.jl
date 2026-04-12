using Test
using AlgebraicOptimization
using LinearAlgebra
using ControlSystems

@testset "UnicycleLQR" begin

    # ── UnicycleModel (VehicleInterface) ─────────────────────────────────────
    @testset "UnicycleModel interface" begin
        model = UnicycleModel(2.0)

        @test state_dim(model)   == 2
        @test control_dim(model) == 1
        @test position_indices(model) == 1:1

        x0, u0 = equilibrium(model)
        @test x0 == zeros(2)
        @test u0 == zeros(1)

        # dynamics at equilibrium should be zero
        @test norm(dynamics(model, x0, u0)) ≈ 0.0  atol=1e-12

        # linearize at equilibrium should give A = [0 v₀; 0 0], B = [0; 1]
        v0 = model.params.v0
        A, B = linearize(model)
        @test A ≈ [0.0 v0; 0.0 0.0]  atol=1e-10
        @test B ≈ reshape([0.0; 1.0], 2, 1)  atol=1e-10
    end

    # ── Coupling scales with speed ────────────────────────────────────────────
    @testset "coupling scales with v0" begin
        for v0 in [0.0, 1.0, 3.0, 5.0]
            A, _ = linearize(UnicycleModel(v0))
            @test A[1, 2] ≈ v0  atol=1e-10   # the key coupling term
        end
        # at v0=0 the plant is decoupled
        A_zero, _ = linearize(UnicycleModel(0.0))
        @test A_zero ≈ zeros(2, 2)  atol=1e-10
    end

    # ── Controllability ───────────────────────────────────────────────────────
    @testset "controllability" begin
        for v0 in [1.0, 2.0, 5.0]
            A, B = linearize(UnicycleModel(v0))
            @test rank(ctrb(A, B)) == 2
        end
        # at v0=0 the position state is uncontrollable
        A0, B0 = linearize(UnicycleModel(0.0))
        @test rank(ctrb(A0, B0)) == 1
    end

    # ── LinearLQRController ───────────────────────────────────────────────────
    @testset "LinearLQRController" begin
        model = UnicycleModel(2.0)
        Q = Diagonal([1.0, 0.1])
        R = reshape([1.0], 1, 1)
        ctrl = LinearLQRController(model; Q=Q, R=R)

        @test size(ctrl.K) == (1, 2)
        @test ctrl.x_eq == zeros(2)
        @test ctrl.u_eq == zeros(1)

        # closed-loop must be stable
        A, B = linearize(model)
        @test all(real.(eigvals(A - B * ctrl.K)) .< 0)

        # at equilibrium, control output must be zero
        u = compute_control(ctrl, zeros(2), zeros(2))
        @test u ≈ zeros(1)  atol=1e-12

        # K[1,2] > 0: heading error should produce a corrective yaw rate
        @test ctrl.K[1, 2] > 0
    end

    # ── UnicyclePIDController ─────────────────────────────────────────────────
    @testset "UnicyclePIDController" begin
        model = UnicycleModel(2.0)
        ctrl  = UnicyclePIDController(model; ω_c=1.5)

        @test ctrl.Kp > 0
        @test ctrl.Kd > 0

        # at equilibrium, control output must be zero
        u = compute_control(ctrl, zeros(2), zeros(2))
        @test u ≈ zeros(1)  atol=1e-12
    end

    # ── Simulation convergence ────────────────────────────────────────────────
    @testset "LQR convergence" begin
        model = UnicycleModel(2.0)
        ctrl  = LinearLQRController(model; Q=Diagonal([1.0, 0.1]), R=reshape([1.0], 1, 1))
        rec   = run_unicycle_sim(model, ctrl; x0=[1.0, 0.3], t_end=15.0)

        # lateral and heading error should converge to near zero
        @test abs(rec.x[1, end]) < 0.05
        @test abs(rec.x[2, end]) < 0.05
    end

    @testset "PID convergence" begin
        model = UnicycleModel(2.0)
        ctrl  = UnicyclePIDController(model; ω_c=1.5)
        rec   = run_unicycle_sim(model, ctrl; x0=[1.0, 0.3], t_end=15.0)

        @test abs(rec.x[1, end]) < 0.05
        @test abs(rec.x[2, end]) < 0.05
    end

    # ── LQR uses heading info; P-only controller ignores it ──────────────────
    @testset "LQR outperforms P-only at high speed" begin
        # A P-only controller on e_y has no access to heading error.
        # It cannot anticipate lateral drift from a heading offset.
        # LQR uses both states and should accumulate far less peak lateral error
        # when starting from a pure heading offset (e_y=0, e_ψ=0.5).
        model = UnicycleModel(4.0)   # high speed amplifies the coupling
        Q = Diagonal([1.0, 0.1])
        R = reshape([1.0], 1, 1)
        lqr_ctrl = LinearLQRController(model; Q=Q, R=R)

        # P-only: ω = -Kp * e_y, ignores e_ψ entirely
        # Pick Kp from the LQR gain so the comparison is fair on lateral gain
        Kp_lateral = lqr_ctrl.K[1, 1]
        p_only = (x, x_ref) -> [-Kp_lateral * (x[1] - x_ref[1])]

        x0 = [0.0, 0.5]   # start on-path but with a heading error
        rec_lqr   = run_unicycle_sim(model, lqr_ctrl; x0=x0, t_end=10.0)
        rec_ponly = run_unicycle_sim(model, p_only;   x0=x0, t_end=10.0)

        # LQR sees heading error and corrects immediately — peak lateral error is smaller
        @test maximum(abs.(rec_lqr.x[1, :])) < maximum(abs.(rec_ponly.x[1, :]))
    end

end
