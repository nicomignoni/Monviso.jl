module Monviso

using JuMP, LinearAlgebra
using MathOptInterface: MathOptInterface as MOI

export prox,
    proximal_gradient,
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

The proximal operator iterate.

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
        @constraint(model, [t; g(y) .+ 0.5(_x .- y)] in norm_cone(1 + length(y)))
        @objective(model, Min, t)
        return (x::AbstractArray) -> begin
            fix.(_x, x)
            optimize!(model)
            value.(y)
        end
    else
        return analytical_prox
    end
end

function residual(F::Function, Π::Function, x::AbstractArray, params...)
    return norm(x .- Π(x .- F(x, params...)))
end

"""
    proximal_gradient(
        F; 
        g = x -> 0,
        y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
        model::Union{Nothing, Model} = nothing,
        norm_cone=MOI.SecondOrderCone,
        analytical_prox = identity
    )

The proximal gradient iterate. 

# Description
Given a constant step-size ``\\chi > 0`` and an initial vector ``\\mathbf{x}_0 \\in 
\\mathbb{R}^n``, the basic ``k``-th iterate of the proximal gradient (PG) algorithm is 
[nemirovskij1983problem](@cite):

```math
\\mathbf{x}_{k+1} = \\text{prox}_{g,\\mathcal{S}}(\\mathbf{x}_k - \\chi 
\\mathbf{F}(\\mathbf{x}_k))
```

where ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. The convergence 
of PG is guaranteed for Lipschitz strongly monotone operators, with monotone constant 
``\\mu > 0`` and Lipshitz constants ``L < +\\infty``, when ``\\chi \\in (0, 2\\mu/L^2)``.
"""
function proximal_gradient(
    F; 
    g = x -> 0,
    y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
    model::Union{Nothing, Model} = nothing,
    norm_cone=MOI.SecondOrderCone,
    analytical_prox = identity
)
    Π = prox(g; y=y, model=model, norm_cone=norm_cone, analytical_prox=analytical_prox)
    function iterate(xk::AbstractArray, χ::Real, params...)   
        xk1 = Π(xk .- χ * F(xk, params...))
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
    \\mathbf{y}_k &= \\text{prox}_{\\mathcal{S}}(\\mathbf{x}_k - \\chi \\mathbf{F}
    (\\mathbf{x}_k)) \\\\
    \\mathbf{x}_{k+1} &= \\text{prox}_{\\mathcal{S}}(\\mathbf{y}_k - \\chi \\mathbf{F}
    (\\mathbf{x}_k))
\\end{align*}
```

where ``g : \\mathbb{R}^n \\to \\mathbb{R}`` is a scalar convex (possibly non-smooth) 
function, while ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. 
The convergence of the EGD algorithm is guaranteed for Lipschitz monotone operators, with 
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
        yk = Π(xk - χ * F(xk, params...))
        xk1 = Π(xk - χ * F(yk, params...))
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
    \\mathbf{y}_{k+1} &= \\text{prox}_{g,\\mathcal{S}}(\\mathbf{x}_k - \\chi \\mathbf{F}
    (\\mathbf{y}_k)) \\\\
    \\mathbf{x}_{k+1} &= \\text{prox}_{g,\\mathcal{S}}(\\mathbf{y}_{k+1} - \\chi 
    \\mathbf{F}(\\mathbf{x}_k))
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
    analytical_prox = identity
)
    Π = prox(g; y=y, model=model, norm_cone=norm_cone, analytical_prox=analytical_prox)
    function iterate(xk::AbstractArray, yk::AbstractArray, χ::Real, params...)
        yk1 = Π(xk - χ * F(yk, params...)) 
        xk1 = Π(xk - χ * F(yk1, params...))
        return xk1, yk1
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

The forward-backward-forward iterate.

# Description
Given a constant step-size ``\\chi > 0`` and an initial vector ``\\mathbf{x}_0 \\in 
\\mathbb{R}^n``, the ``k``-th iterate of Forward-Backward-Forward (FBF) algorithm is 
[tseng2000modified](@cite):

```math 
\\begin{align*}
    \\mathbf{y}_k &= \\text{prox}_{\\mathcal{S}}(\\mathbf{x}_k - \\chi F(\\mathbf{x}_k)) \\\\
    \\mathbf{x}_{k+1} &= \\mathbf{y}_k - \\chi (F(\\mathbf{y}_k) - \\chi F(\\mathbf{x}_k))
\\end{align*}
```

where ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. The convergence 
of the FBF algorithm is guaranteed for Lipschitz monotone operators, with Lipschitz 
constant ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{L}\\right)``.
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
    function iterate(xk::AbstractArray, χ::Real, params...)
        F_xk = F(xk, params...)
        yk = Π(xk .- χ * F_xk)
        xk1 = yk .- χ * (F(yk, params...) .- F_xk)
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

The forward-reflected-backward iterate.

# Description
Given a constant step-size ``\\chi > 0`` and initial vectors ``\\mathbf{x}_1,\\mathbf{x}_0 
\\in \\mathbb{R}^n``, the basic ``k``-th iterate of the Forward-Reflected-Backward (FRB) is 
the following [malitsky2020forward](@cite):

```math
\\mathbf{x}_{k+1} = \\text{prox}_{g,\\mathcal{S}}(\\mathbf{x}_k - \\chi (2\\mathbf{F}
(\\mathbf{x}_k) + \\mathbf{F}(\\mathbf{x}_{k-1})))
```

where ``g : \\mathbb{R}^n \\to \\mathbb{R}`` is a scalar convex (possibly non-smooth) 
function, while ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. 
The convergence of the FRB algorithm is guaranteed for Lipschitz monotone operators, with 
Lipschitz constant ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{2L}\\right)``.
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
    function iterate(xk::AbstractArray, x1k::AbstractArray, χ::Real, params...)
        xk1 = Π(xk .- χ * (2F(xk, params...) .+ F(x1k, params...)))
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

The projected reflected gradient iterate.

# Description
Given a constant step-size ``\\chi > 0`` and initial vectors ``\\mathbf{x}_1,\\mathbf{x}_0 
\\in \\mathbb{R}^n``, the basic ``k``-th iterate of the projected reflected gradient (PRG) 
is the following [malitsky2015projected](@cite):

```math 
\\mathbf{x}_{k+1} = \\text{prox}_{g,\\mathcal{S}}(\\mathbf{x}_k - \\chi 
\\mathbf{F}(2\\mathbf{x}_k - \\mathbf{x}_{k-1})) 
```

where ``g : \\mathbb{R}^n \\to \\mathbb{R}`` is a scalar convex (possibly non-smooth) 
function, while ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. 
The convergence of PRG algorithm is guaranteed for Lipschitz monotone operators, with 
Lipschitz constants ``L < +\\infty``, when ``\\chi \\in (0,(\\sqrt{2} - 1)/L)``. 
Differently from the EGD iteration, the PRGD has the advantage of requiring a single 
proximal operator evaluation.
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
    function iterate(xk::AbstractArray, x1k::AbstractArray, χ::Real, params...)
        xk1 = Π(xk .- χ * F(2xk .- x1k, params...))
        return xk1
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
    \\mathbf{y}_k &= \\text{prox}_{g,\\mathcal{S}}\\left(\\mathbf{x}_k - 
        \\chi \\mathbf{F}(\\mathbf{x}_k) + \\frac{1}{k+1}(\\mathbf{x}_0 - 
        \\mathbf{x}_k)\\right) \\\\
    \\mathbf{x}_{k+1} &= \\text{prox}_{g,\\mathcal{S}}\\left(\\mathbf{x}_k - 
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
        yk = Π(xk .- χ * F(xk, params...) .+ (x0 - xk) ./ (k + 1))
        xk1 = Π(xk .- χ * F(yk, params...) .+ (x0 - xk) ./ (k + 1))
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

The accelerated reflected gradient iterate.

# Description
Given a constant step-size ``\\chi > 0`` and initial vectors ``\\mathbf{x}_1,\\mathbf{x}_0 
\\in \\mathbb{R}^n``, the basic ``k``-th iterate of the accelerated reflected gradient 
(ARG) is the following [cai2022accelerated](@cite):

```math
\\begin{align*}
    \\mathbf{y}_k &= 2\\mathbf{x}_k - \\mathbf{x}_{k-1} + \\frac{1}{k+1}
    (\\mathbf{x}_0 - \\mathbf{x}_k) - \\frac{1}{k}(\\mathbf{x}_k - 
    \\mathbf{x}_{k-1}) \\\\
    \\mathbf{x}_{k+1} &= \\text{prox}_{g,\\mathcal{S}}\\left(\\mathbf{x}_k - 
        \\chi \\mathbf{F}(\\mathbf{y}_k) + \\frac{1}{k+1}(\\mathbf{x}_0 - 
        \\mathbf{x}_k)\\right)
\\end{align*}
```

where ``g : \\mathbb{R}^n \\to \\mathbb{R}`` is a scalar convex (possibly non-smooth) 
function, while ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n`` is the VI mapping. The
convergence of the ARG algorithm is guaranteed for Lipschitz monotone operators, with 
Lipschitz constant ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{12L}\\right)``.
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
    function iterate(
        xk::AbstractArray, 
        x1k::AbstractArray,
        x0::AbstractArray, 
        k::Int, 
        χ::Real, 
        params...
    )
        yk = 2xk .- x1k .+ (x0 .- xk) ./ (k + 1) .- (xk .- x1k) ./ k
        xk1 = Π(xk .- χ * F(yk, params...) .+ (x0 .- xk) ./ (k + 1))
        return xk1
    end
    return iterate
end

"""
    fast_optimistic_gradient_descent_ascent(
        F; 
        g = x -> 0,
        y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
        model::Union{Nothing, Model} = nothing,
        norm_cone=MOI.SecondOrderCone,
        analytical_prox = identity
    )

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
convergence of the ARG algorithm is guaranteed for Lipschitz monotone operators, with 
Lipschitz constant ``L < +\\infty``, when ``\\chi \\in \\left(0,\\frac{1}{4L}\\right)`` 
and ``\\alpha > 2``.
"""
function fast_optimistic_gradient_descent_ascent(
    F;
    g = x -> 0,
    y::Union{Nothing, AbstractArray{VariableRef}} = nothing,
    model::Union{Nothing, Model} = nothing,
    norm_cone=MOI.SecondOrderCone,
    analytical_prox = identity
)
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
    \\mathbf{x}_{k+1} &= \\text{prox}_{g,\\mathcal{S}}\\left(\\mathbf{y}_k - \\chi\\left(1 
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
        xk1 = Π(yk .- χ * (1 + k / (k + α)) .* F_diff)
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
    \\mathbf{y}_{k+1} &= \\frac{(\\phi - 1)\\mathbf{x}_k + \\phi\\mathbf{y}_k}{\\phi} \\\\
    \\mathbf{x}_{k+1} &= \\text{prox}_{g,\\mathcal{S}}(\\mathbf{y}_{k+1} - \\chi 
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
        xk1 = Π(yk1 .- χ .* F(xk, params...))
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
\\phi^2``, where ``\\phi \\in (0,\\varphi]`` and ``\\varphi = \\frac{1+\\sqrt{5}}{2}`` 
is the golden ratio. 
Moreover, let ``\\bar{\\chi} \\gg 0`` be a constant (arbitrarily large) step-size. 
Given the initial terms ``\\mathbf{x}_0,\\mathbf{x}_1 \\in \\mathbb{R}^n``, ``\\mathbf{y}_0 
= \\mathbf{x}_1``, and ``\\chi_0 > 0``, the ``k``-th iterate for aGRAAL is the following:
 
```math
\\begin{align*} 
\\chi_k &= \\min\\left\\{\\rho\\chi_{k-1},
      \\frac{\\phi\\theta_k \\|\\mathbf{x}_k
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
        sk = min(ρ * s1k, ϕ * tk * norm(xk .- x1k) / (4 * s1k * norm(F_diff)), χ_large)
        yk1 = ((ϕ - 1) .* xk .+ yk) ./ ϕ
        xk1 = Π(yk1 .- sk .* F(xk, params...))
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
Let ``\\theta_0 = 1``, ``\\rho = 1/\\phi + 1/\\phi^2``, where ``\\phi \\in (0,\\varphi]`` 
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
        \\frac{\\phi\\theta_k \\|\\mathbf{x}_k
        -\\mathbf{x}_{k-1}\\|^2}{4\\chi_{k-1}\\|\\mathbf{F}(\\mathbf{x}_k)
        -\\mathbf{F}(\\mathbf{x}_{k-1})\\|^2}, \\bar{\\chi}\\right\\} \\\\
    c_k &= \\left(\\langle J(\\mathbf{x}_k) - J(\\mathbf{x}_{k-1}) > 0 \\rangle 
        \\text{ and } \\langle f_k \\rangle \\right) 
        \\text{ or } \\left\\langle \\min\\{J(\\mathbf{x}_{k-1}), J(\\mathbf{x}_k)\\} < 
        J(\\mathbf{x}_k) + \\frac{1}{\\bar{k}} \\right\\rangle \\\\
    f_k &= \\text{not \$\\langle c_k \\rangle\$} \\\\
    \\bar{k} &= \\begin{cases} \\bar{k}+1 & \\text{if \$c_k\$ is true} \\\\ 
        \\bar{k} & \\text{otherwise} \\end{cases} \\\\
    \\mathbf{y}_{k+1} &= 
        \\begin{cases}
            \\dfrac{(\\phi - 1)\\mathbf{x}_k + \\phi\\mathbf{y}_k}{\\phi} & 
            \\text{if \$c_k\$ is true} \\\\
            \\mathbf{x}_k & \\text{otherwise}
        \\end{cases} \\\\
    \\mathbf{x}_{k+1} &= \\text{prox}_{g,\\mathcal{S}}(\\mathbf{y}_{k+1} - \\chi_k 
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
        sk = min(ρ * s1k, ϕ * tk * norm(xk .- x1k) / (4 * s1k * norm(F_diff)), χ_large)
        Jk = residual(F, Π, xk, params...)
        J1k = residual(F, Π, x1k, params...)
        condition = (Jk - J1k > 0 && flag) || (min(Jk, J1k) < Jk + 1 / ck)
        yk1 = condition ? (ϕ - 1) .* xk .+ yk ./ ϕ : xk
        flag = !flag
        ck1 = condition ? ck + 1 : ck
        xk1 = Π(yk1 .- sk .* F(xk, params...))
        tk1 = ϕ * sk / s1k
        return xk1, yk1, sk, tk1, ck1
    end
    return iterate
end

end # module Monviso
