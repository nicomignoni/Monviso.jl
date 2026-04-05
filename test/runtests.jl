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
    Π = prox(y=y_vector, model=model_vector)
    @test Π(x_vector) isa Vector 

    # Differnent norm
    Π = prox(y=y_vector, model=model_vector, norm_cone=MOI.NormOneCone)
    @test Π(x_vector) isa Vector
end

@testset "Projection, ComponentVectors" begin

    # With a constraint set
    Π = prox(y=y_cv, model=model_cv)
    @test Π(x_cv) isa ComponentVector

    # Differnent norm
    Π = prox(y=y_cv, model=model_cv, norm_cone=MOI.NormOneCone)
    @test Π(x_cv) isa ComponentVector
end

@testset "Projected gradient, Vector" begin
    H = rand(n, n)
    H = H' * H # make positive semidefinite
    F(x) = H * x

    χ = 0.01
    pg_iterate = proximal_gradient(F; y=y_vector, model=model_vector)
    @test pg_iterate(x_vector, χ) isa Vector
end

@testset "Forward-backward-forward, Vector" begin
    H = rand(n, n)
    H = H' * H # make positive semidefinite
    F(x) = H * x

    χ = 0.01
    fbf_iterate = forward_backward_forward(F; y=y_vector, model=model_vector)
    @test fbf_iterate(x_vector, χ) isa Vector
end

@testset "Closure-based later iterates" begin
    H = rand(n, n)
    H = H' * H
    F(x) = H * x

    χ = 0.01
    x0 = rand(n)
    x1 = rand(n)
    y0 = rand(n)
    z0 = zeros(n)

    fogda_iterate = fast_optimistic_gradient_descent_ascent(F)
    x_fogda, y_fogda = fogda_iterate(x0, x1, y0, 1, χ)
    @test x_fogda isa Vector
    @test y_fogda isa Vector

    cfogda_iterate = constrained_fast_optimistic_gradient_descent_ascent(F; y=y_vector, model=model_vector)
    x_cfogda, y_cfogda, z_cfogda = cfogda_iterate(x0, x1, y0, z0, 1, χ)
    @test x_cfogda isa Vector
    @test y_cfogda isa Vector
    @test z_cfogda isa Vector

    graal_iterate = golden_ratio_algorithm(F; y=y_vector, model=model_vector)
    x_graal, y_graal = graal_iterate(x0, y0, χ)
    @test x_graal isa Vector
    @test y_graal isa Vector

    agraal_iterate = adaptive_golden_ratio_algorithm(F; y=y_vector, model=model_vector)
    x_agraal, y_agraal, s_agraal, t_agraal = agraal_iterate(x0, x1, y0, χ)
    @test x_agraal isa Vector
    @test y_agraal isa Vector
    @test s_agraal isa Real
    @test t_agraal isa Real

    hgraal_iterate = hybrid_golden_ratio_algorithm_1(F; y=y_vector, model=model_vector)
    x_hgraal, y_hgraal, s_hgraal, t_hgraal, c_hgraal = hgraal_iterate(x0, x1, y0, χ, 1.0, 1)
    @test x_hgraal isa Vector
    @test y_hgraal isa Vector
    @test s_hgraal isa Real
    @test t_hgraal isa Real
    @test c_hgraal isa Int
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
#=p = prox(x, set, optimizer)=#
#=p(x)=#

#=# Projected gradient=#
#=F(x) = 2x=#
#=χ = 0.1=#
#==#
#=proximal_gradient_iterate = ProxGradient(x, F, set, optimizer)=#
#=proximal_gradient_iterate(x, χ)=#
