# Feasibility problem

Let us consider $M$ balls in $\mathbb{R}^n$, where the $i$-th ball of radius $r_i > 0$ centered in $\mathbf{c}_i \in \mathbb{R}^n$ is given by $\mathcal{B}_i(\mathbf{c}_i, r_i) \subset \mathbb{R}^n$. We are interested in finding a point belonging to their intersection, i.e., we want to solve the following

```math
\begin{equation} 
    \text{find} \ \mathbf{x} \ \text{subject to} \ \mathbf{x} \in \bigcap_{i = 1}^M \mathcal{B}_i(\mathbf{c}_i, r_i)
\end{equation}
```

It is straightforward to verify that the projection of a point onto $\mathcal{B}_i(\mathbf{c}_i,r_i)$ is evaluated as

```math
\begin{equation}
    \mathbf{P}_i(\mathbf{x}) := 
    \text{proj}_{\mathcal{B}_i(\mathbf{c}_i,r_i)}(\mathbf{x}) = 
    \begin{cases}
        \displaystyle r_i\frac{\mathbf{x} - 
        \mathbf{c}_i}{\|\mathbf{x} - \mathbf{c}_i\|} & \text{if} \ \|\mathbf{x} - \mathbf{c}_i\| > r_i \\
        \mathbf{x} & \text{otherwise}
    \end{cases}
\end{equation}
```

Due to the non-expansiveness of the projection in $(2)$, one can find a solution for $(1)$ as the fixed point of the following iterate 

```math
\begin{equation}
    \mathbf{x}_{k+1} = \mathbf{T}(\mathbf{x}_k) = \frac{1}{M}\sum_{i = 1}^M\mathbf{P}_i(\mathbf{x}_k) 
\end{equation}
```

which result from the well-known Krasnoselskii-Mann iterate. By letting $\mathbf{F} = \mathbf{I} - \mathbf{T}$, where $\mathbf{I}$ denotes the identity operator, the fixed point for $(3)$ can be treated as the canonical VI [bauschke1996projection](@cite). 

Let's start by creating the problems' data.

```@example FEASIBILITY
using Random, LinearAlgebra
using Monviso, Distributions

Random.seed!(2024)

# Problem data
n, M = 10, 50
centers = rand(Normal(0, 100), (n, M))
radii = rand(Normal(30, 100), M) .|> abs

nothing # hide
```

Then, let's create the projection operator $\mathbf{P}(\cdot)$ and the VI mapping $\mathbf{F}(\cdot)$.

```@example FEASIBILITY
P(x) = let ds = x .- centers, ms = map(norm, eachcol(x .- centers))
    reduce(hcat, [m > r ? r .* d ./ m : x for (m, d, r) in zip(ms, eachcol(ds), radii)])
end
T(x) = mean(P(x), dims=2)
F(x) = x .- T(x)

nothing # hide
```

Finally, let's instantiate the iterate functions. Note that $\mathcal{X} = \mathbb{R}^n$, so the proximal operator is the identity itself. 

```@example FEASIBILITY
frb_iter = forward_reflected_backward(F)
prg_iter = projected_reflected_gradient(F)
eag_iter = extra_anchored_gradient(F)

nothing # hide
```

The projection $\mathbf{P}(\cdot)$ is merely monotone, so $\mathbf{F}(\cdot)$ is merely monotone as well. As examples, we can use the forward-reflected-backward ([`Monviso.forward_reflected_backward`](@ref)), the projected reflected gradient ([`Monviso.projected_reflected_gradient`](@ref)), and the extra anchored gradient ([`Monviso.extra_anchored_gradient`](@ref)).

```@example FEASIBILITY
max_iter = 200
step_size = 0.01

residuals = (
    frb = Vector{Float32}(undef, max_iter),
    prg = Vector{Float32}(undef, max_iter),
    eag = Vector{Float32}(undef, max_iter)
)

# Initial solution and iterative process 
x0 = rand(n)

let sol_frb = (xk = x0, x1k = x0)
    sol_prg = (xk = x0, x1k = x0)
    sol_eag = (xk = x0,)

    for k in 1:max_iter
        # Forward-reflected-backward
        xk1_frb = frb_iter(sol_frb..., step_size)
        residuals.frb[k] = norm(xk1_frb .- sol_frb.xk)
        sol_frb = (xk = xk1_frb, x1k = sol_frb.xk)
    
        # Projected reflected gradient
        xk1_prg = prg_iter(sol_prg..., step_size)
        residuals.prg[k] = norm(xk1_prg .- sol_prg.xk)
        sol_prg = (xk = xk1_prg, x1k = sol_prg.xk)
    
        # Extra anchored gradient
        xk1_eag = eag_iter(sol_eag..., x0, k, step_size)
        residuals.eag[k] = norm(xk1_eag .- sol_eag.xk)
        sol_eag = (xk = xk1_eag,)
    end
end

nothing # hide
```

The API docs have some detail on the [convention for naming iterative steps](../api.md). Let's check out the residuals for each method.

```@example FEASIBILITY
using Plots, LaTeXStrings

plot(
    1:max_iter, [residuals.frb, residuals.prg, residuals.eag];
    xlabel=L"Iterations ($k$)",
    ylabel=L"$||\mathbf{x}_k - \mathbf{x}_{k+1}||$",
    labels=[
        "Forward reflected backward"
        "Projected-reflected-backward"
        "Extra anchored gradient"
    ] |> permutedims,
    linewidth=2
)
```
