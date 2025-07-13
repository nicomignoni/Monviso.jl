using Test, Monviso, JuMP, Clarabel, ComponentArrays
using MathOptInterface: MathOptInterface as MOI

n, m = 30, 20
A = rand(m, n)
b = rand(m)

vector_set = (
    (model, x) -> @constraint(model, x .>= 0),
    (model, x) -> @constraint(model, A * x .+ b .<= 0)
)

componentvector_set = (
    (model, x) -> @constraint(model, x .>= 0),
    (model, x) -> @constraint(model, A * x .+ b .<= 0),
    (model, x) -> @constraint(model, x.x .+ x.z .- sum(x.y) .>= 0)
)

@testset "Projection, Vector" begin
    x = 5rand(n)
    x_func(model) = @variable(model, [1:n])

    # With a constraint set
    Π = proj(Clarabel.Optimizer, x_func, vector_set)
    @test Π(x) isa Vector 

    # Unconstrained
    Π = proj(Clarabel.Optimizer, x_func)
    @test Π(x) isa Vector 

    # Differnent norm
    Π = proj(Clarabel.Optimizer, x_func, norm_cone=MOI.NormOneCone)
    @test Π(x) isa Vector
end

@testset "Projection, ComponentVectors" begin
    x = ComponentVector(x=rand(9), y=rand(4, 5), z=5) 
    x_func(model) = ComponentVector(@variable(model, [1:length(x)]), getaxes(x))

    # With a constraint set
    Π = proj(Clarabel.Optimizer, x_func, componentvector_set)
    @test Π(x) isa ComponentVector

    # Unconstrained
    Π = proj(Clarabel.Optimizer, x_func)
    @test Π(x) isa ComponentVector 

    # Differnent norm
    Π = proj(Clarabel.Optimizer, x_func, norm_cone=MOI.NormOneCone)
    @test Π(x) isa ComponentVector
end

@testset "Projected gradient, Vector" begin
    x = 5rand(n)
    x_func(model) = @variable(model, [1:n])

    H = rand(n, n)
    H = H' * H # make positive semidefinite
    F(x) = H * x

    pg = proj_gradient(Clarabel.Optimizer, F, x_func, vector_set)
    χ = 0.01
    @test begin 
        x, _ = pg(x, χ) 
        x isa Vector  
    end
end

#=x = NV(a=rand(2, 3, 4), b=rand(4))=#
#==#
#=set = (=#
#=    (model, z) -> z >= 0,=#
#=    (model, z) - z.a .- 10 <= 0,=#
#=    z -> norm(z.b, 2) <= 0=#
#=)=#
#==#
#=optimizer = Clarabel.Optimizer=#
#==#
#=# Projection=#
#=p = proj(x, set, optimizer)=#
#=p(x)=#

#=# Projected gradient=#
#=F(x) = 2x=#
#=χ = 0.1=#
#==#
#=pg = ProxGradient(x, F, set, optimizer)=#
#=pg(x, χ)=#

