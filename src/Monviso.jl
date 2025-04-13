module Monviso

using JuMP

export proj, pg, fbf

const DESCR_TYPE = Tuple{Vararg{Function}}

proj_variable(model::Model, x::AbstractVector) = @variable(model, [1:length(x)])

"""
The projection operator closure.
"""
function proj(x::AbstractVector, descr::DESCR_TYPE, optimizer; optimize_kwargs...)
    model = Model(optimizer)

    # Create main variable and slack variable for second-order cone constraint (l2-norm)
    _z = proj_variable(model, x)
    _x = @variable(model, [1:length(x)])
    @variable(model, t)

    # Create the constraints, including the l2-norm epigraphic refomulation
    for f in descr
        r = f(model, _z)
        r isa Union{VariableRef,ConstraintRef} ||
            error(
                "The return type for functions in desc must be a VaraibleRef or ConstraintRef,
                got a $(typeof(r)) instead."
            )
    end

    # Objective norm
    @constraint(model, [t; 0.5(_x .- _z)] in SecondOrderCone())
    @objective(model, Min, t)

    return (x::AbstractVector) -> (
        fix.(_x, x); optimize!(model; optimize_kwargs...); value.(_z)
    )
end


"""
The projected gradient closure. 
"""
function pg(x::AbstractVector, F::Function, descr::DESCR_TYPE, optimizer; optimize_kwargs...)
    Π = proj(x::AbstractVector, descr::DESCR_TYPE, optimizer; optimize_kwargs...)
    return (x::AbstractVector, χ::Real) -> Π(x .- χ * F(x))
end

"""
The forward-backward-forward closure.
"""
function fbf(x::AbstractVector, F::Function, descr::DESCR_TYPE, optimizer; optimize_kwargs...)
    Π = proj(x::AbstractVector, descr::DESCR_TYPE, optimizer; optimize_kwargs...)
    return (x::AbstractVector, χ::Real) -> begin
        Fx = F(x)
        x̄ = Π(x .- χ * Fx)
        x̄ - χ * (F(x̄) .- Fx)
    end
end



end # module Monviso
