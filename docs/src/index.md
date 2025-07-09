# Monviso.jl 

*Solving monotone variational inequalities in Julia*

## Quickstart
Let $F(\mathbf{x}) = \mathbf{H} \mathbf{x}$ for some $\mathbf{H} \succ 0$, $g(\mathbf{x}) = \|\mathbf{x}\|_1$, and $\mathcal{S} = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{A} \mathbf{x} \leq \mathbf{b}\}$, for some $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{b} \in \mathbb{R}^n$. It is straightforward to verify that $F(\cdot)$ is strongly monotone with $\mu = \lambda_{\min}(\mathbf{H})$ and Lipschitz with $L = \|\mathbf{H}\|_2$. The solution of the VI in can be implemented using `Monviso.jl` as follows

```@example QUICKSTART
using Monviso, JuMP, Clarabel, LinearAlgebra, Plots

# Create the problem data
const n, m = 30, 40
H = rand(n, n)
A = rand(m, n)
b = rand(m)

H =  H * H'

# Lipschitz and strong monotonicity constants
L = norm(H)
μ = eigvals(H) |> minimum

# Define F, g, and S
var_func = (model) -> @variable(model, [1:n])
F(x) = H * x
set = (
    (model, x) -> @constraint(model, A * x <= b),
    (model, x) -> @constraint(model, x >= 0)
)

# Instantitate the projected gradient method (`pg`) 
pg = proj_gradient(Clarabel.Optimizer, F, var_func, set)

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
