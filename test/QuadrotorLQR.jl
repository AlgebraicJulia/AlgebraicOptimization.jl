using Test
using AlgebraicOptimization
using LinearAlgebra
using ControlSystems

@testset "QuadrotorLQR" begin

    p = DEFAULT_PARAMS

    # ── Hover equilibrium ────────────────────────────────────────────────────
    @testset "hover_equilibrium" begin
        x0, ω0 = hover_equilibrium(p)
        @test length(x0) == 12
        @test all(x0 .== 0.0)
        # Four rotors at ω0 must produce exactly m·g thrust
        @test 4 * p.kf * ω0^2 ≈ p.m * p.g  atol=1e-10
    end

    # ── Linearisation ────────────────────────────────────────────────────────
    @testset "linearize_hover" begin
        A_pos, B_pos, A_att, B_att, A_z, B_z = linearize_hover(p)

        @test size(A_pos) == (4, 4)
        @test size(B_pos) == (4, 2)
        @test size(A_att) == (6, 6)
        @test size(B_att) == (6, 3)
        @test size(A_z)   == (2, 2)
        @test size(B_z)   == (2, 1)

        # Position: ẍ = g·θ, ÿ = −g·φ
        @test A_pos[3, 1] == 0.0
        @test B_pos[3, 2] ≈  p.g
        @test B_pos[4, 1] ≈ -p.g

        # Attitude: kinematics on diagonal, reaction-torque in B
        @test A_att[1, 4] == 1.0   # φ̇ = p
        @test A_att[2, 5] == 1.0   # θ̇ = q
        @test A_att[3, 6] == 1.0   # ψ̇ = r
        @test B_att[4, 1] ≈ p.l / p.Ix    # roll  torque
        @test B_att[5, 2] ≈ p.l / p.Iy    # pitch torque
        @test B_att[6, 3] ≈ 1.0   / p.Iz  # yaw   (reaction torque)

        # Altitude: z̈ = δU₁/m
        @test A_z[1, 2] == 1.0
        @test B_z[2, 1] ≈ 1.0 / p.m
    end

    # ── Controllability ──────────────────────────────────────────────────────
    @testset "controllability" begin
        A_pos, B_pos, A_att, B_att, A_z, B_z = linearize_hover(p)

        @test rank(ctrb(A_pos, B_pos)) == size(A_pos, 1)
        @test rank(ctrb(A_att, B_att)) == size(A_att, 1)
        @test rank(ctrb(A_z,   B_z))   == size(A_z,   1)
    end

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

        A_pos, B_pos, A_att, B_att, A_z, B_z = linearize_hover(p)

        # Each closed-loop must be stable
        Acl_pos = A_pos - B_pos * ctrl.K_pos
        Acl_att = A_att - B_att * ctrl.K_att
        Acl_z   = A_z   - B_z   * ctrl.K_z

        @test all(real.(eigvals(Acl_pos)) .< 0)
        @test all(real.(eigvals(Acl_att)) .< 0)
        @test all(real.(eigvals(Acl_z))   .< 0)
    end

    # ── compute_control ──────────────────────────────────────────────────────
    @testset "compute_control" begin
        ctrl = LQRController(p)

        # At hover with zero error the virtual inputs should equal nominal
        x0, ω0 = hover_equilibrium(p)
        ω², U = compute_control(ctrl, x0, x0)

        @test U[1] ≈ p.m * p.g  atol=1e-8    # total thrust = weight
        @test U[2] ≈ 0.0        atol=1e-8
        @test U[3] ≈ 0.0        atol=1e-8
        @test U[4] ≈ 0.0        atol=1e-8
        @test all(ω² .≥ 0)                    # no negative thrust commands

        # Perturb altitude: controller should command more thrust
        x_high = copy(x0); x_high[3] = 1.0   # z error = +1 m (above ref)
        _, U_high = compute_control(ctrl, x_high, x0)
        @test U_high[1] < p.m * p.g           # reduce thrust to descend
    end

    # ── SheafControllerInterface ─────────────────────────────────────────────
    @testset "SheafControllerInterface" begin
        ctrl  = LQRController(p)
        iface = SheafControllerInterface(ctrl; planner_hz=10.0)

        x0, _ = hover_equilibrium(p)

        # Before any planner update x_ref = 0 (hover)
        ω², U = step!(iface, x0)
        @test U[1] ≈ p.m * p.g  atol=1e-8

        # Planner updates reference (e.g. move 1 m in x)
        x_ref_new = copy(x0); x_ref_new[1] = 1.0
        update_reference!(iface, x_ref_new, 0.1)
        @test iface.x_ref[1] ≈ 1.0

        # Inner loop step with the new reference
        ω², U = step!(iface, x0)
        @test length(ω²) == 4
        @test length(U)  == 4
    end

end
