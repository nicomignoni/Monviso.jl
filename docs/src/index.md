# Monviso.jl 

*Solving monotone variational inequalities in Julia*

## Installation
Install `Monviso.jl` from the Julia REPL

```julia
add ] https://github.com/nicomignoni/Monviso.jl.git
```

## Quickstart
Let $F(\mathbf{x}) = \mathbf{H} \mathbf{x}$ for some $\mathbf{H} \succ 0$, $g(\mathbf{x}) = \|\mathbf{x}\|_1$, and $\mathcal{S} = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{A} \mathbf{x} \leq \mathbf{b}\}$, for some $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{b} \in \mathbb{R}^n$. It is straightforward to verify that $F(\cdot)$ is strongly monotone with $\mu = \lambda_{\min}(\mathbf{H})$ and Lipschitz with $L = \|\mathbf{H}\|_2$. The solution of the VI in can be implemented using `Monviso.jl` as follows

```@example QUICKSTART
using Monviso, JuMP, Clarabel, LinearAlgebra, Plots

# Create the problem data
const n, m = 30, 40
H = rand(n, n)
A = rand(m, n)
b = rand(m)

# Make H positive semidefinite
H =  H * H'

# Lipschitz and strong monotonicity constants
L = norm(H)
μ = eigvals(H) |> minimum

# Define the F mapping
F(x) = H * x

# Define a JuMP.Model
model = Model(Clarabel.Optimizer)
y = @variable(model, [1:n])
@constraint(model, A * y <= b),
@constraint(model, y >= 0)

# Instantitate the projected gradient method (`pg`) 
pg = proj_gradient(F, y, model)

# Define the initial point, step-size, and max number of iterations
x = rand(n) .+ 4
χ = 2/L^2
T = 10

# Solve the VI
residual = zeros(T)
for τ in 1:T
    x⁺ = pg(x, χ)
    residual[τ] = norm(x .- x⁺) 
    x[:] = x⁺
end

# Lets look at the residuals
plot(residual, yscale=:log, xlabel="Iteration (τ)", ylabel="log||x - x⁺||")
```
