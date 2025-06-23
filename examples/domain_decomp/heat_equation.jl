using AlgebraicOptimization
using Decapodes
#using Plots
using CombinatorialSpaces
using GeometryBasics: Point2
using Catlab
using CairoMakie
Point2D = Point2{Float64}

#=dset = DeltaSet1D()

add_vertices!(dset, 20)

for i in 2:20
    add_edge!(dset, i - 1, i)
end=#

function circle(n, c)
    mesh = EmbeddedDeltaSet1D{Bool,Point2D}()
    map(range(0, 2pi - (pi / (2^(n - 1))); step=pi / (2^(n - 1)))) do t
        add_vertex!(mesh, point=Point2D(cos(t), sin(t)) * (c / 2pi))
    end
    add_edges!(mesh, 1:(nv(mesh)-1), 2:nv(mesh))
    add_edge!(mesh, nv(mesh), 1)
    dualmesh = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2D}(mesh)
    subdivide_duals!(dualmesh, Circumcenter())
    mesh, dualmesh
end
mesh, dualmesh = circle(9, 500)


function cover_mesh(partition_function, s)
    vertex_partition = map(partition_function, s[:point])
    parts = map(unique(vertex_partition)) do p
        vp = findall(i -> i == p, vertex_partition)
        sp = non(negate(Subobject(s; V=vp)))
    end
    return parts
end

function pizza_slices(x)
    x[1] > 0 + 2 * x[2] > 0
end
circ_quads = cover_mesh(pizza_slices, dualmesh)
# draw(circ_quads)
circ_quads[1]

function draw(s; color=:blue)
    f = Figure()
    ax = CairoMakie.Axis(f[1, 1])
    scatter!(ax, s, color=color)
    return f, ax
end

function draw(submesh::Subobject; color=:orange)
    ϕ = hom(submesh)
    f, ax = draw(codom(ϕ))
    scatter!(ax, dom(ϕ), color=color)
    f
end

function draw(cover::Vector{T}; color=:orange) where T<:Subobject
    f = Figure()
    n = length(cover)
    for i in 1:n
        for j in i:n
            ax = CairoMakie.Axis(f[i, j])
            ui, uj = cover[i], cover[j]
            ϕ = hom(meet(ui, uj))
            scatter!(ax, codom(ϕ), color=:blue)
            scatter!(ax, dom(ϕ), color=color)
        end
    end
    f
end

draw(circ_quads)




#=
# Define Mesh
function circle(n, c)
    mesh = EmbeddedDeltaSet1D{Bool,Point2D}()
    map(range(0, 2pi - (pi / (2^(n - 1))); step=pi / (2^(n - 1)))) do t
        add_vertex!(mesh, point=Point2D(cos(t), sin(t)) * (c / 2pi))
    end
    add_edges!(mesh, 1:(nv(mesh)-1), 2:nv(mesh))
    add_edge!(mesh, nv(mesh), 1)
    dualmesh = EmbeddedDeltaDualComplex1D{Bool,Float64,Point2D}(mesh)
    subdivide_duals!(dualmesh, Circumcenter())
    mesh, dualmesh
end
mesh, dualmesh = circle(8, 0)

scatter(dualmesh[:point])

function laplacian(dualmesh)
    return -(dec_hodge_star(1, dualmesh) * dec_differential(0, dualmesh) * dec_inv_hodge_star(0, dualmesh) * dec_dual_derivative(0, dualmesh))
end

L = laplacian(dualmesh)
=#



