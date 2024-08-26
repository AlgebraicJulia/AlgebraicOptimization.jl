using Test
using AlgebraicOptimization
using Catlab

seed_random()

d = @relation (x,y,z) begin
    f(w,x)
    g(u,w,y)
    h(u,w,z)
end


# Test flow graph composition
g1 = random_open_flowgraph(10, .2, 2)
g2 = random_open_flowgraph(10, .2, 3)
g3 = random_open_flowgraph(10, .2, 3)

g_comp = oapply(d, [g1, g2, g3])

# Test naturality of MCNF
γ = 0.1
iters = 20
p1 = to_problem(g1)
p2 = to_problem(g2)
p3 = to_problem(g3)
p_comp1 = oapply(d, [p1, p2, p3])

p_comp2 = to_problem(g_comp)

opt1 = Euler(gradient_flow(p_comp1), γ)
opt2 = Euler(gradient_flow(p_comp2), γ)

r11 = simulate(opt1, zeros(length(opt1.S)), iters)
r22 = simulate(opt2, zeros(length(opt2.S)), iters)

@test r11 ≅ r22

o1 = dual_decomposition(g1, 0.1)             # Why the redefinition?
o2 = dual_decomposition(g2, 0.1)
o3 = dual_decomposition(g3, 0.1)

# o1 = Euler(gradient_flow(p1),γ)
# o2 = Euler(gradient_flow(p2),γ)
# o3 = Euler(gradient_flow(p3),γ)

comp_opt1 = oapply(OpenDiscreteOpt(), d, [o1,o2,o3])
comp_opt2 = dual_decomposition(g_comp, γ)

res1 = simulate(comp_opt1, zeros(length(comp_opt1.S)), iters)
res2 = simulate(comp_opt2, zeros(length(comp_opt2.S)), iters)

@test res1 ≅ res2
@test r11 ≅ res1   # Why are we checking this?

