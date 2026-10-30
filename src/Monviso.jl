module Monviso

using JuMP, LinearAlgebra
using MathOptInterface: MathOptInterface as MOI

export prox,
    forward_backward,
    extragradient,
    popov,
    forward_backward_forward,
    forward_reflected_backward,
    projected_reflected_gradient,
    extra_anchored_gradient,
    accelerated_reflected_gradient,
    fast_optimistic_gradient_descent_ascent,
    constrained_fast_optimistic_gradient_descent_ascent,
    golden_ratio_algorithm,
    adaptive_golden_ratio_algorithm,
    hybrid_golden_ratio_algorithm_1

const GOLDEN_RATIO = 0.5(1 +  sqrt(5))

include("docstrings.jl")

"""
    prox(
        y::AbstractArray{VariableRef},
        model::Model;
        g=nothing,
        norm_cone=MOI.SecondOrderCone,
    )

Construct a proximal operator.

For the default `MOI.SecondOrderCone`, the returned closure evaluates
``\\operatorname{prox}_{\\chi g + \\iota_{\\mathcal S}}(x)``. Call it as
`Π(x, χ)`; `Π(x)` remains shorthand for `χ = 1`. With a different
`norm_cone`, it minimizes the corresponding squared cone norm instead, so
the Euclidean convergence statements of the algorithms below do not apply
without a separate analysis.

# Arguments
$DOCS_Y
$DOCS_MODEL

# Keywords
$DOCS_G
$DOCS_NORM_CONE
"""
function prox(
    g=x -> 0;
    y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
    model::Union{Nothing, Model} = nothing,
    norm_cone=MOI.SecondOrderCone,
    analytical_prox = identity
)
    if y !== nothing && model !== nothing
        _x = @variable(model, [1:length(y)])
        t = @variable(model)
        @constraint(model, [t; _x .- y] in norm_cone(1 + length(y)))
        return (x::AbstractArray, χ::Real = 1) -> begin
            fix.(_x, x)
            @objective(model, Min, χ * g(y) + 0.5 * t^2)
            optimize!(model)
            value.(y)
        end
    else
        return (x::AbstractArray, χ::Real = 1) ->
            applicable(analytical_prox, x, χ) ? analytical_prox(x, χ) : analytical_prox(x)
    end
end

function residual(F::Function, Π::Function, x::AbstractArray, params...)
    return norm(x .- Π(x .- F(x, params...)))
end

"""
    forward_backward(A, R_B)

The forward backward iterate.

# Description
The forward-backward [passty1979ergodic](@cite) iteration is

```math
\\begin{align*}
    \\mathbf{x}_{k+1} = R_{\\mathbf{B}}(\\mathbf{x}_k - \\chi \\mathbf{A}(\\mathbf{x}_k))
\\end{align*}
```

Convergence is guaranteed for Lipschitz strongly monotone ``\\mathbf{A}``, 
with monotone constant ``\\mu > 0`` and Lipshitz constants 
``L < +\\infty``, when ``\\chi \\in (0, 2\\mu/L^2)``.
This assumes that the supplied `R_B` is the stepsize-dependent resolvent for
the current `χ`, that the backward operator is maximally monotone, and that a
solution exists.
"""
function forward_backward(A, R_B)
    function iterate(xk::AbstractArray, χ::Real, A_args = (), R_B_args = ())  
        xk1 = R_B(xk .- χ * A(xk, A_args...), R_B_args...)
        return xk1
    end 
    return iterate
end

"""
    forward_backward(
        F; 
        g = x -> 0,
        y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
        model::Union{Nothing, Model} = nothing,
        norm_cone=MOI.SecondOrderCone,
        analytical_prox = identity
    )

The VI-oriented forward backward iterate. 

# Description
The forward-backward [nemirovskij1983problem](@cite) iteration is

```math
\\begin{align*}
    \\mathbf{x}_{k+1} = \\text{prox}_{\\chi g,\\mathcal{S}}(\\mathbf{x}_k - \\chi 
    \\mathbf{F}(\\mathbf{x}_k))
\\end{align*}
```

Convergence is guaranteed for Lipschitz strongly monotone operators, with monotone constant
``\\mu > 0`` and Lipshitz constant ``L < +\\infty``, when 
``\\chi \\in (0, 2\\mu/L^2)``.
"""
function forward_backward(
    F; 
    g = x -> 0,
    y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
    model::Union{Nothing, Model} = nothing,
    norm_cone=MOI.SecondOrderCone,
    analytical_prox = identity
)
    Π = prox(g; y=y, model=model, norm_cone=norm_cone, analytical_prox=analytical_prox)
    function iterate(xk::AbstractArray, χ::Real, F_args = (), _ = ())
        return Π(xk .- χ * F(xk, F_args...), χ)
    end
    return iterate
end

"""
    forward_backward_forward(A, R_B)

The forward-backward-forward iterate.

# Description
The forward-backward-forward [tseng2000modified](@cite) iteration is

```math 
\\begin{align*}
    \\mathbf{y}_k &= R_{\\mathbf{B}}(\\mathbf{x}_k - \\chi \\mathbf{A}(\\mathbf{x}_k)) \\\\
    \\mathbf{x}_{k+1} &= \\mathbf{y}_k - 
    \\chi (\\mathbf{A}(\\mathbf{y}_k) - \\mathbf{A}(\\mathbf{x}_k))
\\end{align*}
```

Convergence is guaranteed for Lipschitz monotone ``\\mathbf{A}``, with Lipschitz 
constant ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{L}\\right)``.
"""
function forward_backward_forward(A, R_B)
    function iterate(xk::AbstractArray, χ::Real, A_args = (), R_B_args = ())
        A_xk = A(xk, A_args...)
        yk = R_B(xk .- χ * A_xk, R_B_args...)
        xk1 = yk .- χ * (A(yk, A_args...) .- A_xk)
        return xk1
    end
    return iterate
end

"""
    forward_backward_forward(
        F; 
        g = x -> 0,
        y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
        model::Union{Nothing, Model} = nothing,
        norm_cone=MOI.SecondOrderCone,
        analytical_prox = identity
    )

The VI-oriented forward-backward-forward iterate.

# Description
The forward-backward-forward [tseng2000modified](@cite) iteration is

```math 
\\begin{align*}
    \\mathbf{y}_k &= \\text{prox}_{\\chi g,\\mathcal{S}}(\\mathbf{x}_k - \\chi 
    \\mathbf{F}(\\mathbf{x}_k)) \\\\
    \\mathbf{x}_{k+1} &= \\mathbf{y}_k - \\chi (\\mathbf{F}(\\mathbf{y}_k) - 
    \\mathbf{F}(\\mathbf{x}_k))
\\end{align*}
```

Convergence is guaranteed for Lipschitz monotone operators, with Lipschitz constant
``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{L}\\right)``.
"""
function forward_backward_forward(
    F; 
    g = x -> 0,
    y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
    model::Union{Nothing, Model} = nothing,
    norm_cone=MOI.SecondOrderCone,
    analytical_prox = identity
)
    Π = prox(g; y=y, model=model, norm_cone=norm_cone, analytical_prox=analytical_prox)
    function iterate(xk::AbstractArray, χ::Real, F_args = (), _ = ())
        F_xk = F(xk, F_args...)
        yk = Π(xk .- χ * F_xk, χ)
        return yk .- χ * (F(yk, F_args...) .- F_xk)
    end
    return iterate
end

"""
    forward_reflected_backward(A, R_B)

The forward-reflected-backward iterate.

# Description
The forward-reflected-backward [malitsky2020forward](@cite) iteration is

```math
\\mathbf{x}_{k+1} = R_{\\chi\\mathbf{B}}(\\mathbf{x}_k - 2\\chi \\mathbf{A}
(\\mathbf{x}_k) + \\chi\\mathbf{A}(\\mathbf{x}_{k-1}))
```

Convergence of the FRB algorithm is guaranteed for Lipschitz monotone operators, with 
Lipschitz constant ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{2L}\\right)``.
"""
function forward_reflected_backward(A, R_B)
    function iterate(xk::AbstractArray, x1k::AbstractArray, χ::Real, A_args = (), R_B_args = ())
        xk1 = R_B(xk .- 2χ * A(xk, A_args...) .+ χ * A(x1k, A_args...), R_B_args...)
        return xk1
    end
    return iterate
end

"""
    forward_reflected_backward(
        F; 
        g = x -> 0,
        y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
        model::Union{Nothing, Model} = nothing,
        norm_cone=MOI.SecondOrderCone,
        analytical_prox = identity
    )

The VI-oriented forward-reflected-backward iterate.

# Description
The forward-reflected-backward [malitsky2020forward](@cite) iteration is

```math
\\mathbf{x}_{k+1} = \\text{prox}_{\\chi g,\\mathcal{S}}(\\mathbf{x}_k - 2\\chi \\mathbf{F}
(\\mathbf{x}_k) + \\chi\\mathbf{F}(\\mathbf{x}_{k-1}))
```

Convergence is guaranteed for Lipschitz monotone operators, with Lipschitz constant
``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{2L}\\right)``.
"""
function forward_reflected_backward(
    F; 
    g = x -> 0,
    y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
    model::Union{Nothing, Model} = nothing,
    norm_cone=MOI.SecondOrderCone,
    analytical_prox = identity
)
    Π = prox(g; y=y, model=model, norm_cone=norm_cone, analytical_prox=analytical_prox)
    function iterate(xk::AbstractArray, x1k::AbstractArray, χ::Real, F_args = (), _ = ())
        return Π(xk .- 2χ * F(xk, F_args...) .+ χ * F(x1k, F_args...), χ)
    end
    return iterate
end

"""
    projected_reflected_gradient(A, R_B)

The projected reflected gradient iterate.

# Description
The projected-reflected-backward [malitsky2015projected](@cite) iteration is

```math 
\\begin{equation*}
    \\mathbf{x}_{k+1} = R_{\\mathbf{B}}(\\mathbf{x}_k - \\chi 
    \\mathbf{A}(2\\mathbf{x}_k - \\mathbf{x}_{k-1})) 
\\end{equation*}
```

Convergence is guaranteed for Lipschitz monotone ``\\mathbf{A}``, with 
Lipschitz constants ``L < +\\infty``, when ``\\chi \\in (0,(\\sqrt{2} - 1)/L)``. 
"""
function projected_reflected_gradient(A, R_B)
    function iterate(xk::AbstractArray, x1k::AbstractArray, χ::Real, A_args = (), R_B_args = ())
        xk1 = R_B(xk .- χ * A(2xk .- x1k, A_args...), R_B_args...)
        return xk1
    end
    return iterate
end

"""
    projected_reflected_gradient(
        F; 
        g = x -> 0,
        y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
        model::Union{Nothing, Model} = nothing,
        norm_cone=MOI.SecondOrderCone,
        analytical_prox = identity
    )

The VI-oriented projected reflected gradient iterate.

# Description
The projected-reflected-backward [malitsky2015projected](@cite) iteration is

```math 
\\mathbf{x}_{k+1} = \\text{prox}_{\\chi g,\\mathcal{S}}(\\mathbf{x}_k - \\chi 
\\mathbf{F}(2\\mathbf{x}_k - \\mathbf{x}_{k-1})) 
```

Convergence is guaranteed for Lipschitz monotone operators, with Lipschitz constant
``L < +\\infty``, when ``\\chi \\in (0,(\\sqrt{2} - 1)/L)``. 
"""
function projected_reflected_gradient(
    F; 
    g = x -> 0,
    y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
    model::Union{Nothing, Model} = nothing,
    norm_cone=MOI.SecondOrderCone,
    analytical_prox = identity
)
    Π = prox(g; y=y, model=model, norm_cone=norm_cone, analytical_prox=analytical_prox)
    function iterate(xk::AbstractArray, x1k::AbstractArray, χ::Real, F_args = (), _ = ())
        return Π(xk .- χ * F(2xk .- x1k, F_args...), χ)
    end
    return iterate
end

"""
    accelerated_reflected_gradient(A, R_B)

The accelerated reflected gradient iterate.

# Description
The accelerated reflected gradient [cai2022accelerated](@cite) iteration is

```math
\\begin{align*}
    \\mathbf{y}_k &= 2\\mathbf{x}_k - \\mathbf{x}_{k-1} + \\frac{1}{k+1}
    (\\mathbf{x}_0 - \\mathbf{x}_k) - \\frac{1}{k}(\\mathbf{x}_0 - 
    \\mathbf{x}_{k-1}) \\\\
    \\mathbf{x}_{k+1} &= R_{\\mathbf{B}}\\left(\\mathbf{x}_k - 
        \\chi \\mathbf{A}(\\mathbf{y}_k) + \\frac{1}{k+1}(\\mathbf{x}_0 - 
        \\mathbf{x}_k)\\right)
\\end{align*}
```

Convergence of the ARG algorithm is guaranteed for Lipschitz monotone operators, with 
Lipschitz constant ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{12L}\\right)``.
"""
function accelerated_reflected_gradient(A, R_B)
    function iterate(
        xk::AbstractArray, 
        x1k::AbstractArray,
        x0::AbstractArray, 
        k::Int, 
        χ::Real,
        A_args = (),
        R_B_args = (),
    )
        yk = 2xk .- x1k .+ (x0 .- xk) ./ (k + 1) .- (x0 .- x1k) ./ k
        xk1 = R_B(xk .- χ * A(yk, A_args...) .+ (x0 .- xk) ./ (k + 1), R_B_args...)
        return xk1
    end
    return iterate
end

"""
    accelerated_reflected_gradient(
        F; 
        g = x -> 0,
        y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
        model::Union{Nothing, Model} = nothing,
        norm_cone=MOI.SecondOrderCone,
        analytical_prox = identity
    )

The VI-oriented accelerated reflected gradient iterate.

# Description
The accelerated reflected gradient [cai2022accelerated](@cite) iteration is

```math
\\begin{align*}
    \\mathbf{y}_k &= 2\\mathbf{x}_k - \\mathbf{x}_{k-1} + \\frac{1}{k+1}
    (\\mathbf{x}_0 - \\mathbf{x}_k) - \\frac{1}{k}(\\mathbf{x}_0 - 
    \\mathbf{x}_{k-1}) \\\\
    \\mathbf{x}_{k+1} &= \\text{prox}_{\\chi g,\\mathcal{S}}\\left(\\mathbf{x}_k - 
        \\chi \\mathbf{F}(\\mathbf{y}_k) + \\frac{1}{k+1}(\\mathbf{x}_0 - 
        \\mathbf{x}_k)\\right)
\\end{align*}
```

Convergence is guaranteed for Lipschitz monotone operators, with Lipschitz constant
``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{12L}\\right)``.
"""
function accelerated_reflected_gradient(
    F; 
    g = x -> 0,
    y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
    model::Union{Nothing, Model} = nothing,
    norm_cone=MOI.SecondOrderCone,
    analytical_prox = identity
)
    Π = prox(g; y=y, model=model, norm_cone=norm_cone, analytical_prox=analytical_prox)
    function iterate(xk::AbstractArray, x1k::AbstractArray, x0::AbstractArray, k::Int, χ::Real, F_args = (), _ = ())
        yk = 2xk .- x1k .+ (x0 .- xk) ./ (k + 1) .- (x0 .- x1k) ./ k
        return Π(xk .- χ * F(yk, F_args...) .+ (x0 .- xk) ./ (k + 1), χ)
    end
    return iterate
end

"""
    peaceman_rachford(R_A, R_B)

The Peaceman-Rachford iterate

# Description
The Peaceman-Rachford iteration is given by

```math
\\begin{align*}
    \\mathbf{y}_k &= 2R_{\\mathbf{B}}(\\mathbf{x}_k) - \\mathbf{x}_k
    \\matbf{x}_{k+1} &= 2R_{\\mathbf{A}}(\\mathbf{y}_k) - \\mathbf{y}_k
\\end{align*}
```
"""
function peaceman_rachford(R_A, R_B)
    function iterate(xk::AbstractVector, R_A_args = (), R_B_args = ())
        yk1 = 2R_B(xk, R_B_args...) .- xk
        xk1 = 2R_A(yk1, R_A_args...) .- yk1
        return xk1
    end
    return iterate
end

"""
    douglas_rachford(R_A, R_B)

# Description
The Douglas-Rachford iteration is given by

```math
\\begin{align*}
    \\mathbf{y}_k &= 2R_{\\mathbf{B}}(\\mathbf{x}_k) - \\mathbf{x}_k
    \\matbf{x}_{k+1} &= \\mathbf{x}_k + R_{\\mathbf{A}}(\\mathbf{y}_k) - \\mathbf{y}_k
\\end{align*}
```
"""
function douglas_rachford(R_A, R_B)
    function iterate(xk::AbstractVector, R_A_args = (), R_B_args = ())
        yk1 = 2R_B(xk, R_B_args...) .- xk
        xk1 = xk .+ R_A(yk1, R_A_args...) .- yk1
        return xk1 
    end
    return iterate
end

"""
    extragradient(
        F; 
        g = x -> 0,
        y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
        model::Union{Nothing, Model} = nothing,
        norm_cone=MOI.SecondOrderCone,
        analytical_prox = identity
    )

The extragradient iterate

# Description
Given a constant step-size ``\\chi > 0`` and an initial vector ``\\mathbf{x}_0 \\in 
\\mathbb{R}^n``, the ``k``-th iterate of the extragradient algorithm (EG) is 
[korpelevich1976extragradient](@cite):

```math
\\begin{align*}
    \\mathbf{y}_k &= \\text{prox}_{\\chi g,\\mathcal{S}}(\\mathbf{x}_k - \\chi \\mathbf{F}
    (\\mathbf{x}_k)) \\\\
    \\mathbf{x}_{k+1} &= \\text{prox}_{\\chi g,\\mathcal{S}}(\\mathbf{x}_k - \\chi \\mathbf{F}
    (\\mathbf{y}_k))
\\end{align*}
```

where ``g : \\mathbb{R}^n \\to \\mathbb{R}`` is a scalar convex (possibly non-smooth) 
function, while ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. 
The convergence of the EG algorithm is guaranteed for Lipschitz monotone operators, with 
Lipschitz constant ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{L}\\right)``.
"""
function extragradient(
    F; 
    g = x -> 0,
    y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
    model::Union{Nothing, Model} = nothing,
    norm_cone=MOI.SecondOrderCone,
    analytical_prox = identity
)
    Π = prox(g; y=y, model=model, norm_cone=norm_cone, analytical_prox=analytical_prox)
    function iterate(xk::AbstractArray, χ::Real, params...)
        yk = Π(xk - χ * F(xk, params...), χ)
        xk1 = Π(xk - χ * F(yk, params...), χ)
        return xk1
    end
    return iterate
end

"""
    popov(
        F; 
        g = x -> 0,
        y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
        model::Union{Nothing, Model} = nothing,
        norm_cone=MOI.SecondOrderCone,
        analytical_prox = identity
    )

The Popov's method iterate

# Description
Given a constant step-size ``\\chi > 0`` and an initial vectors 
``\\mathbf{x}_0,\\mathbf{y}_0 \\in \\mathbb{R}^n``, the ``k``-th iterate of Popov's Method 
(PM) is [popov1980modification](@cite):

```math
\\begin{align*}
    \\mathbf{y}_{k+1} &= \\text{prox}_{\\chi g,\\mathcal{S}}(\\mathbf{x}_k - \\chi \\mathbf{F}
    (\\mathbf{y}_k)) \\\\
    \\mathbf{x}_{k+1} &= \\text{prox}_{\\chi g,\\mathcal{S}}(\\mathbf{x}_k - \\chi 
    \\mathbf{F}(\\mathbf{y}_{k+1}))
\\end{align*}
```

where ``g : \\mathbb{R}^n \\to \\mathbb{R}`` is a scalar convex (possibly non-smooth) 
function, while ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. The 
convergence of PM is guaranteed for Lipschitz monotone operators, with Lipschitz constant 
``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{2L}\\right)``.
"""
function popov(
    F; 
    g = x -> 0,
    y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
    model::Union{Nothing, Model} = nothing,
    norm_cone=MOI.SecondOrderCone,
    analytical_prox=identity
)
    Π = prox(g; y=y, model=model, norm_cone=norm_cone, analytical_prox=analytical_prox)
    function iterate(xk::AbstractArray, yk::AbstractArray, χ::Real, params...)
        yk1 = Π(xk - χ * F(yk, params...), χ)
        xk1 = Π(xk - χ * F(yk1, params...), χ)
        return xk1, yk1
    end
    return iterate
end

"""
    extra_anchored_gradient(
        F; 
        g = x -> 0,
        y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
        model::Union{Nothing, Model} = nothing,
        norm_cone=MOI.SecondOrderCone,
        analytical_prox = identity
    )

The extra-anchored gradient iterate.

# Description
Given a constant step-size ``\\chi > 0`` and an initial vector ``\\mathbf{x}_0 \\in 
\\mathbb{R}^n``, the ``k``-th  iterate of extra anchored gradient (EAG) algorithm is 
[yoon2021accelerated](@cite):

```math
\\begin{align*}
    \\mathbf{y}_k &= \\text{prox}_{\\chi g,\\mathcal{S}}\\left(\\mathbf{x}_k - 
        \\chi \\mathbf{F}(\\mathbf{x}_k) + \\frac{1}{k+1}(\\mathbf{x}_0 - 
        \\mathbf{x}_k)\\right) \\\\
    \\mathbf{x}_{k+1} &= \\text{prox}_{\\chi g,\\mathcal{S}}\\left(\\mathbf{x}_k - 
        \\chi \\mathbf{F}(\\mathbf{y}_k) + \\frac{1}{k+1}(\\mathbf{x}_0 - 
        \\mathbf{x}_k)\\right)
\\end{align*}
```

where ``g : \\mathbb{R}^n \\to \\mathbb{R}`` is a scalar convex  (possibly non-smooth) 
function, while ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. 
The convergence of the EAG algorithm is guaranteed for Lipschitz monotone operators, with 
Lipschitz constant ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{\\sqrt{3}L} 
\\right)``.
"""
function extra_anchored_gradient(
    F; 
    g = x -> 0,
    y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
    model::Union{Nothing, Model} = nothing,
    norm_cone=MOI.SecondOrderCone,
    analytical_prox = identity
)
    Π = prox(g; y=y, model=model, norm_cone=norm_cone, analytical_prox=analytical_prox)
    function iterate(xk::AbstractArray, x0::AbstractArray, k::Int, χ::Real, params...)
        yk = Π(xk .- χ * F(xk, params...) .+ (x0 - xk) ./ (k + 1), χ)
        xk1 = Π(xk .- χ * F(yk, params...) .+ (x0 - xk) ./ (k + 1), χ)
        return xk1
    end
    return iterate
end

"""
    fast_optimistic_gradient_descent_ascent(F)

(Explicit) fast optimistic gradient descent-ascent iterate

# Description
Given a constant step-size ``\\chi > 0`` and initial vectors ``\\mathbf{x}_1,\\mathbf{x}_0,
\\mathbf{y}_0 \\in \\mathbb{R}^n``, the basic ``k``-th iterate of the explicit fast OGDA 
(FOGDA) is the following [boct2025fast](@cite):

```math
\\begin{align*}
    \\mathbf{y}_k &= \\mathbf{x}_k + \\frac{k}{k+\\alpha}(\\mathbf{x}_k - 
        \\mathbf{x}_{k-1}) - \\chi \\frac{\\alpha}{k+\\alpha}
        \\mathbf{F}(\\mathbf{y}_{k-1}) \\\\
    \\mathbf{x}_{k+1} &= \\mathbf{y}_k - \\chi \\frac{2k+\\alpha}
        {k+\\alpha} (\\mathbf{F}(\\mathbf{y}_k) - \\mathbf{F}(\\mathbf{y}_{k-1}))
\\end{align*}
```

where ``g : \\mathbb{R}^n \\to \\mathbb{R}`` is a scalar convex (possibly non-smooth) 
function, while ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. The 
convergence of the FOGDA algorithm is guaranteed for Lipschitz monotone operators, with 
Lipschitz constant ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{4L}\\right)`` 
and ``\\alpha > 2``.
"""
function fast_optimistic_gradient_descent_ascent(F)
    function iterate(
        xk::AbstractArray, 
        x1k::AbstractArray,
        y1k::AbstractArray, 
        k::Int, 
        χ::Real, 
        params...;
        α::Real=2.1,
    )
        yk = xk .+ k .* (xk .- x1k) ./ (k + α) .- χ * α .* F(y1k, params...) ./ (k + α)
        xk1 = yk .- χ * (2k + α) .* (F(yk, params...) .- F(y1k, params...)) ./ (k + α)
        return xk1, yk
    end
    return iterate
end

"""
    constrained_fast_optimistic_gradient_descent_ascent(
        F;
        g = x -> 0,
        y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
        model::Union{Nothing, Model} = nothing,
        norm_cone=MOI.SecondOrderCone,
        analytical_prox = identity
    )

Constrained fast optimistic gradient descent-ascent iterate

# Description
Given a constant step-size ``\\chi > 0`` and initial vectors ``\\mathbf{x}_1 \\in 
\\mathcal{S}``, ``\\mathbf{z}_1 \\in N_{\\mathcal{S}}(\\mathbf{x}_1)``, ``\\mathbf{x}_0,
\\mathbf{y}_0 \\in \\mathbb{R}^n``, the basic ``k``-th iterate of Constrained Fast 
Optimistic Gradient Descent Ascent (CFOGDA) is the following [sedlmayer2023fast](@cite):

```math 
\\begin{align*}
    \\mathbf{y}_k &= \\mathbf{x}_k + \\frac{k}{k+\\alpha}(\\mathbf{x}_k - 
    \\mathbf{x}_{k-1}) - \\chi \\frac{\\alpha}{k+\\alpha}(\\mathbf{F}(\\mathbf{y}_{k-1}) + 
    \\mathbf{z}_k) \\\\
    \\mathbf{x}_{k+1} &= \\text{prox}_{\\chi g,\\mathcal{S}}\\left(\\mathbf{y}_k - \\chi\\left(1 
    + \\frac{k}{k+\\alpha}\\right)(\\mathbf{F}(\\mathbf{y}_k) - \\mathbf{F}
    (\\mathbf{y}_{k-1}) - \\zeta_k)\\right) \\\\
    \\mathbf{z}_{k+1} &= \\frac{k+\\alpha}{\\chi (2k+\\alpha)}( \\mathbf{y}_k - 
    \\mathbf{x}_{k+1}) - (\\mathbf{F}(\\mathbf{y}_k) - \\mathbf{F}(\\mathbf{y}_{k-1}) - 
    \\zeta_k)
\\end{align*}
```

where ``g : \\mathbb{R}^n \\to \\mathbb{R}`` is a scalar convex (possibly non-smooth) 
function, while ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. The 
convergence of the CFOGDA algorithm is guaranteed for Lipschitz monotone operators, with 
Lipschitz constant ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{4L}\\right)`` 
and ``\\alpha > 2``.
This is a projected method: the stated guarantee is for the normal cone of
the constraint set. It does not extend to an arbitrary additional nonzero
`g` merely by replacing the projection with a proximal map.
"""
function constrained_fast_optimistic_gradient_descent_ascent(
    F;
    g = x -> 0,
    y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
    model::Union{Nothing, Model} = nothing,
    norm_cone=MOI.SecondOrderCone,
    analytical_prox = identity
)
    Π = prox(g; y=y, model=model, norm_cone=norm_cone, analytical_prox=analytical_prox)
    function iterate(
        xk::AbstractArray,
        x1k::AbstractArray,
        y1k::AbstractArray,
        zk::AbstractArray,
        k::Int,
        χ::Real,
        params...;
        α::Real=2.1,
    )
        F_y1k = F(y1k, params...)
        yk = xk .+ k .* (xk .- x1k) ./ (k + α) .- χ * α .* (F_y1k .+ zk) ./ (k + α)
        F_diff = F(yk, params...) .- F_y1k .- zk
        xk1 = Π(yk .- χ * (1 + k / (k + α)) .* F_diff, χ)
        zk1 = (k + α) .* (yk .- xk1) ./ (χ * (2k + α)) .- F_diff
        return xk1, yk, zk1
    end
    return iterate
end

"""
    golden_ratio_algorithm(
        F;
        g = x -> 0,
        y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
        model::Union{Nothing, Model} = nothing,
        norm_cone=MOI.SecondOrderCone,
        analytical_prox = identity
    )

Golden ratio algorithm iterate

# Description
Given a constant step-size ``\\chi > 0`` and initial vectors ``\\mathbf{x}_0,\\mathbf{y}_0 
\\in \\mathbb{R}^n``, the basic ``k``-th iterate the golden ratio algorithm (GRAAL) is the 
following [malitsky2020golden](@cite):

```math 
\\begin{align*}
    \\mathbf{y}_{k+1} &= \\frac{(\\phi - 1)\\mathbf{x}_k + \\mathbf{y}_k}{\\phi} \\\\
    \\mathbf{x}_{k+1} &= \\text{prox}_{\\chi g,\\mathcal{S}}(\\mathbf{y}_{k+1} - \\chi 
    \\mathbf{F}(\\mathbf{x}_k))
\\end{align*}
```

The convergence of GRAAL algorithm is guaranteed for Lipschitz monotone operators, with 
Lipschitz constants ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{\\varphi}{2L}
\\right]`` 
and ``\\phi \\in (1,\\varphi]``, where ``\\varphi = \\frac{1+\\sqrt{5}}{2}`` is the golden 
ratio.
"""
function golden_ratio_algorithm(
    F;
    g = x -> 0,
    y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
    model::Union{Nothing, Model} = nothing,
    norm_cone=MOI.SecondOrderCone,
    analytical_prox = identity
)
    Π = prox(g; y=y, model=model, norm_cone=norm_cone, analytical_prox=analytical_prox)
    function iterate(
        xk::AbstractArray,
        yk::AbstractArray,
        χ::Real,
        params...;
        ϕ::Real=GOLDEN_RATIO,
    )
        yk1 = ((ϕ - 1) .* xk .+ yk) ./ ϕ
        xk1 = Π(yk1 .- χ .* F(xk, params...), χ)
        return xk1, yk1
    end
    return iterate
end

"""
    adaptive_golden_ratio_algorithm(
        F;
        g = x -> 0,
        y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
        model::Union{Nothing, Model} = nothing,
        norm_cone=MOI.SecondOrderCone,
        analytical_prox = identity
    )

Adaptive golden ratio algorithm

# Description
The Adaptive Golden Ratio Algorithm (aGRAAL) algorithm is a variation of the Golden Ratio 
Algorithm, with adaptive step size. 
Following [malitsky2020golden](@cite), let ``\\theta_0 = 1``, ``\\rho = 1/\\phi + 1/
\\phi^2``, where ``\\phi \\in (1,\\varphi]`` and ``\\varphi = \\frac{1+\\sqrt{5}}{2}`` 
is the golden ratio. 
Moreover, let ``\\bar{\\chi} \\gg 0`` be a constant (arbitrarily large) step-size. 
Given the initial terms ``\\mathbf{x}_0,\\mathbf{x}_1 \\in \\mathbb{R}^n``, ``\\mathbf{y}_0 
= \\mathbf{x}_1``, and ``\\chi_0 > 0``, the ``k``-th iterate for aGRAAL is the following:
 
```math
\\begin{align*} 
\\chi_k &= \\min\\left\\{\\rho\\chi_{k-1},
      \\frac{\\phi\\theta_{k-1} \\|\\mathbf{x}_k
      -\\mathbf{x}_{k-1}\\|^2}{4\\chi_{k-1}\\|\\mathbf{F}(\\mathbf{x}_k)
      -\\mathbf{F}(\\mathbf{x}_{k-1})\\|^2}, \\bar{\\chi}\\right\\} \\\\
\\mathbf{x}_{k+1}, \\mathbf{y}_{k+1} &= \\texttt{golden\\_ratio\\_algorithm}(\\mathbf{x}_k, \\mathbf{y}_k, 
\\chi_k, \\phi) \\\\
\\theta_{k+1} &= \\phi\\frac{\\chi_k}{\\chi_{k-1}} 
\\end{align*}
```

The convergence guarantees discussed for GRAAL also hold for aGRAAL.
"""
function adaptive_golden_ratio_algorithm(
    F;
    g = x -> 0,
    y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
    model::Union{Nothing, Model} = nothing,
    norm_cone=MOI.SecondOrderCone,
    analytical_prox = identity
)
    Π = prox(g; y=y, model=model, norm_cone=norm_cone, analytical_prox=analytical_prox)
    function iterate(
        xk::AbstractArray,
        x1k::AbstractArray,
        yk::AbstractArray,
        s1k::Real,
        tk::Real=1,
        params...;
        ϕ::Real=GOLDEN_RATIO,
        χ_large::Real=1e6,
    )
        ρ = 1 / ϕ + 1 / ϕ^2
        F_diff = F(xk, params...) .- F(x1k, params...)
        ratio = iszero(norm(F_diff)) ? Inf : norm(xk .- x1k)^2 / norm(F_diff)^2
        sk = min(ρ * s1k, ϕ * tk * ratio / (4 * s1k), χ_large)
        yk1 = ((ϕ - 1) .* xk .+ yk) ./ ϕ
        xk1 = Π(yk1 .- sk .* F(xk, params...), sk)
        tk1 = ϕ * sk / s1k
        return xk1, yk1, sk, tk1
    end
    return iterate
end

"""
    hybrid_golden_ratio_algorithm_1(
        F;
        g = x -> 0,
        y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
        model::Union{Nothing, Model} = nothing,
        norm_cone=MOI.SecondOrderCone,
        analytical_prox = identity
    )

Hybrid golden ratio algorithm I 

# Description
The HGRAAL-1 algorithm [baghbadorani2024hybrid](@cite) is a variation of the Adaptive 
Golden Ratio Algorithm. 
Let ``\\theta_0 = 1``, ``\\rho = 1/\\phi + 1/\\phi^2``, where ``\\phi \\in (1,\\varphi]`` 
and ``\\varphi = \\frac{1+\\sqrt{5}}{2}`` is the golden ratio. 
The residual at point ``\\mathbf{x}_k`` is given by ``J : \\mathbb{R}^n \\to \\mathbb{R}``, 
defined as follows:

```math
J(\\mathbf{x}_k) = \\|\\mathbf{x}_k - \\text{prox}_{g,\\mathcal{S}} (\\mathbf{x}_k - 
\\mathbf{F}(\\mathbf{x}_k))\\|
```

Moreover, let ``\\bar{\\chi} \\gg 0`` be a constant (arbitrarily large) step-size. 
Given the initial terms ``\\mathbf{x}_0,\\mathbf{x}_1 \\in\\mathbb{R}^n``, ``\\mathbf{y}_0 
= \\mathbf{x}_1``, and ``\\chi_0 > 0``, the ``k``-th iterate for HGRAAL-1 is the following:

```math 
\\begin{align*}
    \\chi_k &= \\min\\left\\{\\rho\\chi_{k-1},
        \\frac{\\phi\\theta_{k-1} \\|\\mathbf{x}_k
        -\\mathbf{x}_{k-1}\\|^2}{4\\chi_{k-1}\\|\\mathbf{F}(\\mathbf{x}_k)
        -\\mathbf{F}(\\mathbf{x}_{k-1})\\|^2}, \\bar{\\chi}\\right\\} \\\\
    c_k &= \\left(J(\\mathbf{x}_k) - J(\\mathbf{x}_{k-1}) > 0 \\text{ and } f_k\\right) 
        \\text{ or } \\left(\\min_{0 \\leq i < k} J(\\mathbf{x}_i) < 
        J(\\mathbf{x}_k) + \\frac{1}{\\bar{k}}\\right) \\\\
    f_{k+1} &= \\neg c_k \\\\
    \\bar{k}_{k+1} &= \\begin{cases} \\bar{k}_k & \\text{if } c_k \\text{ is true} \\\\ 
        \\bar{k}_k + 1 & \\text{otherwise} \\end{cases} \\\\
    \\mathbf{y}_{k+1} &= 
        \\begin{cases}
            \\dfrac{(\\phi - 1)\\mathbf{x}_k + \\mathbf{y}_k}{\\phi} & 
            \\text{if } c_k \\text{ is true} \\\\ 
            \\mathbf{x}_k & \\text{otherwise}
        \\end{cases} \\\\
    \\mathbf{x}_{k+1} &= \\text{prox}_{\\chi_k g,\\mathcal{S}}(\\mathbf{y}_{k+1} - \\chi_k 
        \\mathbf{F}(\\mathbf{x}_k)) \\\\
    \\theta_{k+1} &= \\phi\\frac{\\chi_k}{\\chi_{k-1}} 
\\end{align*}
```
"""
function hybrid_golden_ratio_algorithm_1(
    F;
    g = x -> 0,
    y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
    model::Union{Nothing, Model} = nothing,
    norm_cone=MOI.SecondOrderCone,
    analytical_prox = identity
)
    Π = prox(g; y=y, model=model, norm_cone=norm_cone, analytical_prox=analytical_prox)
    flag = false
    best_residual = Inf
    function iterate(
        xk::AbstractArray,
        x1k::AbstractArray,
        yk::AbstractArray,
        s1k::Real,
        tk::Real,
        ck::Int,
        params...;
        ϕ::Real=GOLDEN_RATIO,
        χ_large::Real=1e6,
    )
        ρ = 1 / ϕ + 1 / ϕ^2
        F_diff = F(xk, params...) .- F(x1k, params...)
        ratio = iszero(norm(F_diff)) ? Inf : norm(xk .- x1k)^2 / norm(F_diff)^2
        sk = min(ρ * s1k, ϕ * tk * ratio / (4 * s1k), χ_large)
        Jk = residual(F, Π, xk, params...)
        J1k = residual(F, Π, x1k, params...)
        best_residual = min(best_residual, J1k)
        condition = (Jk - J1k > 0 && flag) || (best_residual < Jk + 1 / ck)
        yk1 = condition ? ((ϕ - 1) .* xk .+ yk) ./ ϕ : xk
        flag = !condition
        ck1 = condition ? ck : ck + 1
        best_residual = min(best_residual, Jk)
        xk1 = Π(yk1 .- sk .* F(xk, params...), sk)
        tk1 = ϕ * sk / s1k
        return xk1, yk1, sk, tk1, ck1
    end
    return iterate
end

end # module Monviso
