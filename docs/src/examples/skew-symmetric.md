# Skew-symmetric operator

A simple example of monotone operator that is not (even locally) strongly monotone is the skewed-symmetric operator [bauschke2020convex](@cite), ``F : \mathbb{R}^{MN} \to \mathbb{R}^{MN}``, which is described as follows

```math
\begin{equation}
    F(\mathbf{x}) = \begin{bmatrix} \mathbf{A}_1 & & \\ & \ddots & \\ & & \mathbf{A}_M \end{bmatrix} \mathbf{x}
\end{equation}
```

for a given ``M \in \mathbb{N}``, where ``\mathbf{A}_i = \text{tril}(\mathbf{B}_i) - \text{triu}(\mathbf{B}_i)``, for some arbitrary ``0 \preceq \mathbf{B}_i \in \mathbb{R}^{N \times N}``, for all ``i = 1, \dots, M``. Let's implement it by first creating the problem data

```@example SKEW
using Random, LinearAlgebra
using Monviso, BlockDiagonals, JuMP, Clarabel

Random.seed!(2024)

M, N = 20, 10

Bs = [let X = rand(N, N); X' * X; end for _ in 1:M]
A = BlockDiagonal([tril(B) .- triu(B) for B in Bs])

nothing # hide
```

Then, let's define $\mathbf{F}(\cdot)$ with it's Lipschitz constant $L$

```@example SKEW
F(x) = A * x
L = norm(A)

nothing # hide
```

Finally, let's create the VI problem and an initial solution

```@example SKEW
model = Model(Clarabel.Optimizer)
set_silent(model)
@variable(model, y[1:N*M])

vi = VI(F; y=y, model=model)

x0 = rand(N * M)

nothing # hide
```

The vector map $\mathbf{F}(\cdot)$ is merely monotone; we can use and compare three different algorithms: fast optimistic gradient descent-ascent ([`Monviso.fogda`](@ref)), golden ratio ([`Monviso.graal`](@ref)), adaptive golden ratio ([`Monviso.agraal`](@ref))

```@example SKEW
GOLDEN_RATIO = 0.5(1 + sqrt(5))

max_iter = 200

residuals = (
    fogda = Vector{Float32}(undef, max_iter),
    graal = Vector{Float32}(undef, max_iter),
    agraal = Vector{Float32}(undef, max_iter)
)

# Copy the same initial solution for the three methods
let sol_fogda = (xk = x0, x1k = x0, y1k = x0),
    sol_graal = (xk = x0, yk = x0),
    sol_agraal = (xk = x0, x1k = rand(N * M), yk = x0, s1k = GOLDEN_RATIO/(2L), tk = 1)
    
    for k in 1:max_iter
        # Fast Optimistic Gradient Descent Ascent 
        xk1_fogda, yk_fogda = fogda(vi, sol_fogda..., k, 1/(4L))
        residuals.fogda[k] = norm(xk1_fogda - sol_fogda.xk)
        sol_fogda = (xk = xk1_fogda, x1k = sol_fogda.xk, y1k = yk_fogda)
    
        # Golden ratio algorithm
        xk1_graal, yk1_graal = graal(vi, sol_graal..., GOLDEN_RATIO/(2L))
        residuals.graal[k] = norm(xk1_graal - sol_graal.xk)
        sol_graal = (xk = xk1_graal, yk = yk1_graal)
    
        # Adaptive golden ratio algorithm
        xk1_agraal, yk1_agraal, sk_agraal, tk1_agraal = agraal(vi, sol_agraal...)
        residuals.agraal[k] = norm(xk1_agraal - sol_agraal.xk)
        sol_agraal = (xk = xk1_agraal, x1k = sol_agraal.xk, yk = yk1_agraal, s1k = sk_agraal, tk = tk1_agraal)
    end
end

nothing # hide
```

The API docs have some detail on the [convention for naming iterative steps](../api.md). Let's check out the residuals for each method. 

```@example SKEW

using Plots, LaTeXStrings

plot(
    1:max_iter, [residuals.fogda, residuals.graal, residuals.agraal];
    xlabel=L"Iterations ($k$)",
    ylabel=L"$||\mathbf{x}_k - \mathbf{x}_{k+1}||$",
    labels=[
        "Fast Optimistic Gradient Descent Ascent"
        "Golden ratio algorithm"
        "Adaptive golden ratio algorithm"
    ] |> permutedims,
    linewidth=2
)
```





