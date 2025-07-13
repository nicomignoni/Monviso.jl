module Monviso

using JuMP, LinearAlgebra
using MathOptInterface: MathOptInterface as MOI

export proj, proj_gradient, forward_backward_forward

const default_set = ((model, x) -> nothing,)
const default_eval_func = (x⁺, x) -> norm(x - x⁺)

# Common arguments and keywords to doc
const DOCS_OPTIMIZER = "- `optimizer` - the optimizer used to solve the projection." 
const DOCS_F = "- `F::Function` - a function of the form `(x, params...) -> AbstractVector` of the same lenght of `x`, i.e., the VI mapping ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n``. Term `params` collects optional arguments that characterize `F` and might change at each iteration."
const DOCS_VAR = "- `x_func` - a function of the type `(model) -> AbstractVector{VariableRef}`, returning a container of `JuMP` variables."
const DOCS_SET = "- `set=default_set` - a `Tuple` of functions of the type `(model, x) -> @constraint(model, expr(x))`, describing the onto which project." 
const DOCS_EVAL_FUNC = "- `eval_func::Function=default_eval_func` - a function of the form `(x⁺, x=nothing) -> Any`, used to evaluate the state of convergence of the iterate, where `x⁺` and `x` are the new and previous iteration's results." 
const DOCS_NORM_CONE = "- `norm_cone::DataType=MOI.SecondOrderCone` - the cone related to the norm characterizing the projection." 
const DOCS_ANALYTICAL_PROJ = "- `analytical_proj::Union{Nothing, Function}=nothing` - the analytical form of the projection of the given set. If provided, it replaces of `proj`." 
const DOCS_SILENT = "- `silent::Bool=true` - verbosity level for the projection solver."

"""
    proj(optimizer, x_func::Function, set::SetType=default_set; norm_cone::DataType=MOI.SecondOrderCone, silent::Bool=true)

The projection operator closure.

# Arguments
$DOCS_OPTIMIZER
$DOCS_VAR
$DOCS_SET

# Keywords
$DOCS_NORM_CONE
$DOCS_SILENT
"""
function proj(
    optimizer,
    x_func,
    set=default_set;
    norm_cone::DataType=MOI.SecondOrderCone,
    silent::Bool=true
)
    model = Model(optimizer)
    if silent; set_silent(model); end

    # Create main variable (_z) and parameter (_x)
    _z = x_func(model)
    _x = @variable(model, [1:length(_z)])

    # Create the constraints set 
    for func in set
        func(model, _z)
    end 

    # Objective norm
    @variable(model, t)
    @constraint(model, [t; 0.5(_x .- _z)] in norm_cone(1 + length(_z)))
    @objective(model, Min, t)

    return (x::AbstractVector) -> begin
        fix.(_x, x)
        optimize!(model)
        value.(_z)
    end
end

# Creates the projection function depending on whether an analytical projection is provided 
function get_projection_func(optimizer, x_func, set, analytical_proj, norm_cone, silent)
    return analytical_proj === nothing ? proj(optimizer, x_func, set; norm_cone=norm_cone, silent=silent) : analytical_proj
end

"""
    proj_gradient(optimizer, F::Function, x_func::Function, set::SetType=default_set; analytical_proj::Union{Nothing, Function}=nothing, norm_cone::DataType=MOI.SecondOrderCone, silent::Bool=true)

The projected gradient closure. 

# Description
Given a constant step-size ``\\chi > 0`` and an initial vector ``\\mathbf{x}_0 \\in \\mathbb{R}^n``, the basic ``k``-th iterate of the projected gradient (PG) algorithm is [^1]:

```math
\\mathbf{x}_{k+1} = \\text{proj}_{\\mathcal{S}}(\\mathbf{x}_k - \\chi \\mathbf{F}(\\mathbf{x}_k))
```

where ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. The convergence of PG is guaranteed for Lipschitz strongly monotone operators, with monotone constant ``\\mu > 0`` and Lipshitz constants ``L < +\\infty``, when ``\\chi \\in (0, 2\\mu/L^2)``.

# Arguments
$DOCS_OPTIMIZER
$DOCS_F
$DOCS_VAR
$DOCS_SET

# Keywords
$DOCS_EVAL_FUNC
$DOCS_ANALYTICAL_PROJ
$DOCS_NORM_CONE
$DOCS_SILENT

# References
[^1] Nemirovskij, A. S., & Yudin, D. B. (1983). Problem complexity and method efficiency in optimization.
"""
function proj_gradient(
    optimizer,
    F,
    x_func,
    set=default_set;
    eval_func=default_eval_func,
    analytical_proj=nothing,
    norm_cone::DataType=MOI.SecondOrderCone,
    silent::Bool=true
)
    Π = get_projection_func(optimizer, x_func, set, analytical_proj, norm_cone, silent)
    return (x::AbstractVector, χ::Real, params...) -> begin
        x⁺ = Π(x .- χ * F(x, params...))

        return x⁺, eval_func(x⁺, x)
    end
end

"""
    forward_backward_forward(optimizer, F::Function, x_func::Function, set::SetType=default_set; analytical_proj::Union{Nothing, Function}=nothing, norm_cone::DataType=MOI.SecondOrderCone, silent::Bool=true)

The forward-backward-forward closure.

# Description
Given a constant step-size ``\\chi > 0`` and an initial vector ``\\mathbf{x}_0 \\in \\mathbb{R}^n``, the ``k``-th iterate of Forward-Backward-Forward (FBF) algorithm is [^1]:

```math 
\\begin{align*}
    \\mathbf{y}_k &= \\text{proj}_{\\mathcal{S}}(\\mathbf{x}_k - \\chi F(\\mathbf{x}_k)) \\\\
    \\mathbf{x}_{k+1} &= \\mathbf{y}_k - \\chi (F(\\mathbf{y}_k) - \\chi F(\\mathbf{x}_k))
\\end{align*}
```

where ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. The convergence of the FBF algorithm is guaranteed for Lipschitz monotone operators, with Lipschitz constant ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{L}\\right)``.

# Arguments
$DOCS_OPTIMIZER
$DOCS_F
$DOCS_VAR
$DOCS_SET

# Keywords
$DOCS_EVAL_FUNC
$DOCS_ANALYTICAL_PROJ
$DOCS_NORM_CONE
$DOCS_SILENT

# References
[^1] Tseng, P. (2000). A modified forward-backward splitting method for maximal monotone mappings. SIAM Journal on Control and Optimization, 38(2), 431-446.
"""
function forward_backward_forward(
    optimizer,
    F,
    x_func,
    set=default_set;
    analytical_proj=nothing,
    norm_cone::DataType=MOI.SecondOrderCone,
    silent::Bool=true
)
    Π = get_projection_func(optimizer, x_func, set, analytical_proj, norm_cone, silent)
    return (x::AbstractVector, χ::Real, params...) -> begin
        x⁺ = Π(x .- χ * F(x, params...))
        x⁺⁺ = x⁺ .- χ * (F(x⁺) .- F(x, params...))

        return x⁺⁺, eval_func(x⁺⁺, x)
    end
end

end # module Monviso
