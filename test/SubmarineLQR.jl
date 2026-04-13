using Test
using AlgebraicOptimization
using LinearAlgebra
using ControlSystems

@testset "SubmarineLQR" begin

    p = DEFAULT_SUBMARINE_PARAMS

    # ── VehicleInterface ──────────────────────────────────────────────────────
    @testset "SubmarineModel interface" begin
        model = SubmarineModel()

        @test state_dim(model)        == 6
        @test control_dim(model)      == 3
        @test position_indices(model) == 1:3

        x0, u0 = equilibrium(model)
        @test x0 == zeros(6)
        @test u0 == zeros(3)

        # dynamics at equilibrium must be zero
        @test norm(dynamics(model, x0, u0)) ≈ 0.0  atol=1e-12
    end

    # ── Linearization ─────────────────────────────────────────────────────────
    @testset "linearize" begin
        model = SubmarineModel()
        A, B = linearize(model)

        @test size(A) == (6, 6)
        @test size(B) == (6, 3)

        mx_eff = p.m + p.mx
        my_eff = p.m + p.my
        mz_eff = p.m + p.mz

        # Kinematics block
        @test A[1, 4] ≈ 1.0  atol=1e-10
        @test A[2, 5] ≈ 1.0  atol=1e-10
        @test A[3, 6] ≈ 1.0  atol=1e-10

        # Quadratic drag linearizes to zero at zero velocity (d/dv(v|v|)|₀ = 0)
        @test A[4, 4] ≈ 0.0  atol=1e-10
        @test A[5, 5] ≈ 0.0  atol=1e-10
        @test A[6, 6] ≈ 0.0  atol=1e-10

        # Input matrix scaled by effective mass per axis
        @test B[4, 1] ≈ 1 / mx_eff  atol=1e-10
        @test B[5, 2] ≈ 1 / my_eff  atol=1e-10
        @test B[6, 3] ≈ 1 / mz_eff  atol=1e-10

        # Axes are decoupled
        @test A[4, 5] ≈ 0.0  atol=1e-10
        @test A[5, 4] ≈ 0.0  atol=1e-10
    end

    # ── Asymmetric effective mass ──────────────────────────────────────────────
    @testset "axis asymmetry" begin
        # Surge: streamlined — low added mass, low drag
        @test p.mx < p.my
        @test p.mx < p.mz
        @test p.dxx < p.dyy
        @test p.dxx < p.dzz
        # Sway/heave symmetric (body of revolution)
        @test p.my  == p.mz
        @test p.dyy == p.dzz
    end

    # ── Controllability ───────────────────────────────────────────────────────
    @testset "controllability" begin
        A, B = linearize(SubmarineModel())
        @test rank(ctrb(A, B)) == 6
    end

    # ── LinearLQRController ───────────────────────────────────────────────────
    @testset "LinearLQRController" begin
        model = SubmarineModel()
        ctrl = LinearLQRController(model)

        @test size(ctrl.K) == (3, 6)
        @test ctrl.x_eq   == zeros(6)
        @test ctrl.u_eq   == zeros(3)

        # Closed-loop must be stable
        A, B = linearize(model)
        @test all(real.(eigvals(A - B * ctrl.K)) .< 0)

        # At equilibrium, control output must be zero
        u = compute_control(ctrl, zeros(6), zeros(6))
        @test u ≈ zeros(3)  atol=1e-12
    end

    # ── LinearizedModel ───────────────────────────────────────────────────────
    @testset "LinearizedModel" begin
        model = SubmarineModel()
        lm = LinearizedModel(model)

        @test state_dim(lm)        == 6
        @test control_dim(lm)      == 3
        @test position_indices(lm) == 1:3

        x0, u0 = equilibrium(lm)
        @test norm(dynamics(lm, x0, u0)) ≈ 0.0  atol=1e-12
    end

    # ── Convergence ───────────────────────────────────────────────────────────
    @testset "LQR convergence" begin
        model = SubmarineModel()
        ctrl = LinearLQRController(model;
            Q = Diagonal([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]),
            R = Diagonal([1.0, 1.0, 1.0]),
        )

        # Linearized model is three pure double integrators — LQR provides all damping
        x0 = [2.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        rec = run_baseline_sim(
            [model], [ctrl], [zeros(6)];
            x0s=[x0], t_end=120.0,
        )

        @test norm(rec[1].x[1:3, end]) < 0.1
        @test norm(rec[1].x[4:6, end]) < 0.1
    end

end
