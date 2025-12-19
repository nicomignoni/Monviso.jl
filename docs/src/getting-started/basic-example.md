# Basic example
Let $F(\mathbf{x}) = \mathbf{H} \mathbf{x} + \text{tanh} |\mathbf{x}|$ for some $\mathbf{H} \succ 0$, $g(\mathbf{x}) = \|\mathbf{x}\|_1$, and $\mathcal{S} = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{A} \mathbf{x} \leq \mathbf{b}\}$, for some $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{b} \in \mathbb{R}^n$. It is straightforward to verify that $F(\cdot)$ is monotone and Lipschitz with $L = \|\mathbf{H}\|_2 + 1$. The solution of the VI in can be implemented using `Monviso.jl` as follows

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
L = norm(H) + 1
μ = eigvals(H) |> minimum

# Define the F mapping
F(x) = H * x .+ tanh.(abs.(x))

# Define a JuMP.Model
model = Model(Clarabel.Optimizer)
y = @variable(model, [1:n])
@constraint(model, A * y <= b),
@constraint(model, y >= 0)

# Instantitate the VI 
vi = VI(F; y=y, model=model)

# Define the initial point, step-size, and max number of iterations
x = rand(n) .+ 4
χ = 2/L^2
T = 10

# Solve the VI using the proximal gradient iterate
residual = zeros(T)
for τ in 1:T
    x⁺ = pg(vi, x, χ)
    residual[τ] = norm(x .- x⁺) 
    x[:] = x⁺
end

# Lets look at the residuals
plot(residual, yscale=:log, xlabel="Iteration (τ)", ylabel="log||x - x⁺||")
```
