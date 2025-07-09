module Monviso

using JuMP
using MathOptInterface: MathOptInterface as MOI

export proj, proj_gradient, forward_backward_forward

const SetType = Tuple{Vararg{Function}}
const default_set = ((model, x) -> nothing,)

"""
The projection operator closure.
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

    # Create main variable and slack variable for second-order cone constraint (l2-norm)
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
The projected gradient closure. 
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
The forward-backward-forward closure.
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
