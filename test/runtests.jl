using Test, Monviso, JuMP, Clarabel, ComponentArrays
using MathOptInterface: MathOptInterface as MOI

n, m = 30, 20
A = rand(m, n)
b = rand(m)

# Model with Vectors
x_vector = 5rand(n)
model_vector = Model(Clarabel.Optimizer)
set_silent(model_vector)
y_vector = @variable(model_vector, [1:n])
@constraint(model_vector, y_vector .>= 0),
@constraint(model_vector, A * y_vector .+ b .<= 0)

# Model with ComponentVectors
x_cv = ComponentVector(x=rand(9), y=rand(4, 5), z=5) 
model_cv = Model(Clarabel.Optimizer)
set_silent(model_cv)
y_cv = ComponentVector(@variable(model_cv, [1:length(x_cv)]), getaxes(x_cv))
@constraint(model_cv, y_cv .>= 0),
@constraint(model_cv, A * y_cv .+ b .<= 0),
@constraint(model_cv, y_cv.x .+ y_cv.z .- sum(y_cv.y) .>= 0)

@testset "Projection, Vector" begin
    # With a constraint set
    Π = proj(y_vector, model_vector)
    @test Π(x_vector) isa Vector 

    # Differnent norm
    Π = proj(y_vector, model_vector, norm_cone=MOI.NormOneCone)
    @test Π(x_vector) isa Vector
end

@testset "Projection, ComponentVectors" begin

    # With a constraint set
    Π = proj(y_cv, model_cv)
    @test Π(x_cv) isa ComponentVector

    # Differnent norm
    Π = proj(y_cv, model_cv, norm_cone=MOI.NormOneCone)
    @test Π(x_cv) isa ComponentVector
end

@testset "Projected gradient, Vector" begin
    H = rand(n, n)
    H = H' * H # make positive semidefinite
    F(x) = H * x

    pg = proj_gradient(F, y_vector, model_vector)
    χ = 0.01
    @test pg(x_vector, χ) isa Vector  
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

