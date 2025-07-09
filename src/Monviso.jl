module Monviso

using JuMP
using MathOptInterface: MathOptInterface as MOI

export proj, proj_gradient, forward_backward_forward

const SetType = Tuple{Vararg{Function}}
const default_set = ((model, x) -> nothing,)

# Common arguments and keywords to doc
const DOCS_OPTIMIZER = "- `optimizer` - the optimizer used to solve the projection." 
const DOCS_F = "- `F::Function` - a function of the form `(x) -> AbstractVector` of the same lenght of `x`, i.e., the VI mapping ``\\mathbf{F} \\to \\mathbb{R}^n \\to \\mathbb{R}^n``."
const DOCS_VAR = "- `var::Function` - a function of the type `(model) -> AbstractVector{VariableRef}`, returning a container of `JuMP` variables."
const DOCS_SET = "- `set::SetType=default_set` - a `Tuple` of functions of the type `(model, x) -> @constraint(model, expr(x))`, describing the onto which project." 
const DOCS_NORM_CONE = "- `norm_cone::DataType=MOI.SecondOrderCone` - the cone related to the norm characterizing the projection." 
const DOCS_ANALYTICAL_PROJ = "- `analytical_proj::Union{Nothing, Function}` - the analytical form of the projection of the given set. If provided, it replaces of `proj`." 
const DOCS_SILENT = "- `silent::Bool=true` - verbosity level for the projection solver."

"""
    proj(optimizer, var::Function, set::SetType=default_set; norm_cone::DataType=MOI.SecondOrderCone, silent::Bool=true)

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
    var::Function,
    set::SetType=default_set;
    norm_cone::DataType=MOI.SecondOrderCone,
    silent::Bool=true
)
    model = Model(optimizer)
    if silent; set_silent(model); end

    # Create main variable (_z) and parameter (_x)
    _z = var(model)
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
function get_projection_func(optimizer, var, set, analytical_proj, norm_cone, silent)
    return analytical_proj === nothing ? proj(optimizer, var, set; norm_cone=norm_cone, silent=silent) : analytical_proj
end

"""
    proj_gradient(optimizer, F::Function, var::Function, set::SetType=default_set; analytical_proj::Union{Nothing, Function}=nothing, norm_cone::DataType=MOI.SecondOrderCone, silent::Bool=true)

The projected gradient closure. 

# Arguments
$DOCS_OPTIMIZER
$DOCS_F
$DOCS_VAR
$DOCS_SET

# Keywords
$DOCS_ANALYTICAL_PROJ
$DOCS_NORM_CONE
$DOCS_SILENT
"""
function proj_gradient(
    optimizer,
    F::Function,
    var::Function,
    set::SetType=default_set;
    analytical_proj::Union{Nothing, Function}=nothing,
    norm_cone::DataType=MOI.SecondOrderCone,
    silent::Bool=true
)
    Π = get_projection_func(optimizer, var, set, analytical_proj, norm_cone, silent)
    return (x::AbstractVector, χ::Real) -> Π(x .- χ * F(x))
end

"""
    forward_backward_forward(optimizer, F::Function, var::Function, set::SetType=default_set; analytical_proj::Union{Nothing, Function}=nothing, norm_cone::DataType=MOI.SecondOrderCone, silent::Bool=true)

The forward-backward-forward closure.

# Arguments
$DOCS_OPTIMIZER
$DOCS_F
$DOCS_VAR
$DOCS_SET

# Keywords
$DOCS_ANALYTICAL_PROJ
$DOCS_NORM_CONE
$DOCS_SILENT
"""
function forward_backward_forward(
    optimizer,
    F::Function,
    var::Function,
    set::SetType=default_set;
    analytical_proj::Union{Nothing, Function}=nothing,
    norm_cone::DataType=MOI.SecondOrderCone,
    silent::Bool=true
)
    Π = get_projection_func(optimizer, var, set, analytical_proj, norm_cone, silent)
    return (x::AbstractVector, χ::Real) -> begin
        x⁺ = Π(x .- χ * F(x))
        x⁺⁺ = x⁺ .- χ * (F(x⁺) .- F(x))
    end
end

end # module Monviso
