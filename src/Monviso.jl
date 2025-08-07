module Monviso

using JuMP, LinearAlgebra
using MathOptInterface: MathOptInterface as MOI

export prox, prox_gradient, forward_backward_forward, extragradient, popov

# Common arguments and keywords to doc
const DOCS_F = "- `F` - a function of the form `(x::AbstractVector, params...) -> AbstractVector` of the same lenght of `x`, i.e., the VI mapping ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n``. Term `params` collects optional arguments that characterize `F` and might change at each iteration."
const DOCS_G = "- `g::Function=nothing` - a function of the form `x::AbstractVector{VariableRef} -> Real`, i.e., the scalar function ``g : \\mathbb{R}^n \\to \\mathbb{R}``."
const DOCS_Y = "- `y::AbstractVector{VariableRef}` - the container of `JuMP.VariableRef` associated to `model`, i.e., ``\\mathbf{y}``."
const DOCS_MODEL = "- `model::Model` - the `JuMP.Model` describing the projection set ``\\mathcal{S}``." 
const DOCS_NORM_CONE = "- `norm_cone=MOI.SecondOrderCone` - the cone related to the norm characterizing the proximal operator." 
const DOCS_ANALYTICAL_PROJ = "- `analytical_prox::Function=nothing` - the analytical form of the proximal operator for the given set. If provided, it replaces of `prox`." 
base_signature(name::String) = "$name(F; y::Union{Nothing, AbstractVector{VariableRef}}=nothing, model::Union{Nothing, Model}=nothing; g::Function=nothing, analytical_prox::Function=nothing, norm_cone=MOI.SecondOrderCone)"

"""
    prox(y::AbstractVector{VariableRef}, model::Model; g=nothing, norm_cone=MOI.SecondOrderCone)

The proximal operator closure.

# Arguments
$DOCS_Y
$DOCS_MODEL

# Keywords
$DOCS_G
$DOCS_NORM_CONE
"""
function prox(
    y::AbstractVector{VariableRef},
    model::Model;
    g=nothing,
    norm_cone=MOI.SecondOrderCone,
)
    g = g === nothing ? (x -> 0) : g
    _x = @variable(model, [1:length(y)])

    # Objective norm
    t = @variable(model)
    @constraint(model, [t; g(y) .+ 0.5(_x .- y)] in norm_cone(1 + length(y)))
    @objective(model, Min, t)

    return (x::AbstractVector) -> begin
        fix.(_x, x)
        optimize!(model)
        value.(y)
    end
end

function get_prox(y, model, analytical_prox, g, norm_cone)
    no_arguments = y === nothing && model === nothing && analytical_prox === nothing
    all_arguments = y !== nothing && model !== nothing && analytical_prox !== nothing
    if no_arguments || all_arguments 
        throw(ArgumentError("Exactly one between (y, model) and analytical_prox must be defined."))
    elseif analytical_prox === nothing 
        return prox(y, model; g=g, norm_cone=norm_cone)
    else 
        return analytical_prox
    end
end

"""
    $(base_signature("prox_gradient"))

The proximal gradient closure. 

# Description
Given a constant step-size ``\\chi > 0`` and an initial vector ``\\mathbf{x}_0 \\in \\mathbb{R}^n``, the basic ``k``-th iterate of the proximal gradient (PG) algorithm is [^1]:

```math
\\mathbf{x}_{k+1} = \\text{prox}_{g,\\mathcal{S}}(\\mathbf{x}_k - \\chi \\mathbf{F}(\\mathbf{x}_k))
```

where ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. The convergence of PG is guaranteed for Lipschitz strongly monotone operators, with monotone constant ``\\mu > 0`` and Lipshitz constants ``L < +\\infty``, when ``\\chi \\in (0, 2\\mu/L^2)``.

[^1]: Nemirovskij, A. S., & Yudin, D. B. (1983). Problem complexity and method efficiency in optimization.

# Arguments
$DOCS_F
$DOCS_Y
$DOCS_MODEL

# Keywords
$DOCS_G
$DOCS_ANALYTICAL_PROJ
$DOCS_NORM_CONE
"""
function prox_gradient(
    F;
    y::Union{Nothing, AbstractVector{VariableRef}}=nothing,
    model::Union{Nothing, Model}=nothing,
    g=nothing,
    analytical_prox=nothing,
    norm_cone=MOI.SecondOrderCone
)   
    Π = get_prox(y, model, analytical_prox, g, norm_cone)
    return (xk::AbstractVector, χ::Real, params...) -> begin
        xk1 = Π(xk .- χ * F(xk, params...))
    end
end

"""
    $(base_signature("forward_backward_forward")) 

The forward-backward-forward closure.

# Description
Given a constant step-size ``\\chi > 0`` and an initial vector ``\\mathbf{x}_0 \\in \\mathbb{R}^n``, the ``k``-th iterate of Forward-Backward-Forward (FBF) algorithm is [^1]:

```math 
\\begin{align*}
    \\mathbf{y}_k &= \\text{prox}_{\\mathcal{S}}(\\mathbf{x}_k - \\chi F(\\mathbf{x}_k)) \\\\
    \\mathbf{x}_{k+1} &= \\mathbf{y}_k - \\chi (F(\\mathbf{y}_k) - \\chi F(\\mathbf{x}_k))
\\end{align*}
```

where ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. The convergence of the FBF algorithm is guaranteed for Lipschitz monotone operators, with Lipschitz constant ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{L}\\right)``.

[^2]: Tseng, P. (2000). A modified forward-backward splitting method for maximal monotone mappings. SIAM Journal on Control and Optimization, 38(2), 431-446.

# Arguments
$DOCS_F
$DOCS_Y
$DOCS_MODEL

# Keywords
$DOCS_G
$DOCS_ANALYTICAL_PROJ
$DOCS_NORM_CONE
"""
function forward_backward_forward(
    F;
    y::Union{Nothing, AbstractVector{VariableRef}}=nothing,
    model::Union{Nothing, Model}=nothing,
    g=nothing,
    analytical_prox=nothing,
    norm_cone=MOI.SecondOrderCone
)
    Π = get_prox(y, model, analytical_prox, g, norm_cone)
    return (xk::AbstractVector, χ::Real, params...) -> begin
        F_xk = F(xk, params...)

        yk = Π(xk .- χ * F_xk)
        xk1 = yk .- χ * (F(yk, params...) .- F_xk)

        return xk1
    end
end

"""
    $(base_signature("extragradient")) 

The extragradient closure

# Description
Given a constant step-size ``\\chi > 0`` and an initial vector ``\\mathbf{x}_0 \\in \\mathbb{R}^n``, the ``k``-th iterate of the extragradient algorithm (EG) is[^1]:

```math
\\begin{align*}
    \\mathbf{y}_k &= \\text{prox}_{\\mathcal{S}}(\\mathbf{x}_k - \\chi \\mathbf{F}(\\mathbf{x}_k)) \\\\
    \\mathbf{x}_{k+1} &= \\text{prox}_{\\mathcal{S}}(\\mathbf{y}_k - \\chi \\mathbf{F}(\\mathbf{x}_k))
\\end{align*}
```

where ``g : \\mathbb{R}^n \\to \\mathbb{R}`` is a scalar convex (possibly non-smooth) function, while ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. The convergence of the EGD algorithm is guaranteed for Lipschitz monotone operators, with Lipschitz constant ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{L}\\right)``.

[^3]: Korpelevich, G. M. (1976). The extragradient method for finding saddle points and other problems. Matecon, 12, 747-756.

# Arguments
$DOCS_F
$DOCS_Y
$DOCS_MODEL

# Keywords
$DOCS_G
$DOCS_ANALYTICAL_PROJ
$DOCS_NORM_CONE
"""
function extragradient(
    F;
    y::Union{Nothing, AbstractVector{VariableRef}}=nothing,
    model::Union{Nothing, Model}=nothing,
    g=nothing,
    analytical_prox=nothing,
    norm_cone=MOI.SecondOrderCone
)
    Π = get_prox(y, model, analytical_prox, g, norm_cone)
    return (xk::AbstractVector, χ::Real, params...) -> begin
        yk = Π(xk - χ * F(xk, params...))
        xk1 = Π(xk - χ * F(yk, params...))

        return xk1
    end
end

"""
    $(base_signature("popov")) 

The Popov's method closure

Given a constant step-size ``\\chi > 0`` and an initial vectors ``\\mathbf{x}_0,\\mathbf{y}_0 \\in \\mathbb{R}^n``, the ``k``-th iterate of Popov's Method (PM) is[^1]:

```math
\\begin{align*}
    \\mathbf{y}_{k+1} &= \\text{prox}_{g,\\mathcal{S}}(\\mathbf{x}_k - \\chi \\mathbf{F}(\\mathbf{y}_k)) \\\\
    \\mathbf{x}_{k+1} &= \\text{prox}_{g,\\mathcal{S}}(\\mathbf{y}_{k+1} - \\chi \\mathbf{F}(\\mathbf{x}_k))
\\end{align*}
```

where ``g : \\mathbb{R}^n \\to \\mathbb{R}`` is a scalar convex (possibly non-smooth) function, while ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. The convergence of PM is guaranteed for Lipschitz monotone operators, with Lipschitz constant ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{2L}\\right)``.

[^4]: Popov, L.D. A modification of the Arrow-Hurwicz method for search of saddle points. Mathematical Notes of the Academy of Sciences of the USSR 28, 845–848 (1980)

# Arguments
$DOCS_F
$DOCS_Y
$DOCS_MODEL

# Keywords
$DOCS_G
$DOCS_ANALYTICAL_PROJ
$DOCS_NORM_CONE
"""
function popov(
    F;
    y::Union{Nothing, AbstractVector{VariableRef}}=nothing,
    model::Union{Nothing, Model}=nothing,
    g=nothing,
    analytical_prox=nothing,
    norm_cone=MOI.SecondOrderCone
)
    Π = get_prox(y, model, analytical_prox, g, norm_cone)
    return (xk::AbstractVector, yk::AbstractVector, χ::Real, params...) -> begin
        yk1 = Π(xk - χ * F(yk, params...)) 
        xk1 = Π(xk - χ * F(yk1, params...))

        return xk1, yk1
    end
end

end # module Monviso
