# Common arguments and keywords to doc
const DOCS_F = "- `F` - a function of the form `(x::AbstractVector, params...) -> AbstractVector` of the same lenght of `x`, i.e., the VI mapping ``\\mathbf{F} : \\mathbb{R}^n \\to \\mathbb{R}^n``. Term `params` collects optional arguments that characterize `F` and might change at each iteration."
const DOCS_G = "- `g::Function=nothing` - a function of the form `x::AbstractVector{VariableRef} -> Real`, i.e., the scalar function ``g : \\mathbb{R}^n \\to \\mathbb{R}``."
const DOCS_Y = "- `y::AbstractVector{VariableRef}` - the container of `JuMP.VariableRef` associated to `model`, i.e., ``\\mathbf{y}``."
const DOCS_MODEL = "- `model::Model` - the `JuMP.Model` describing the projection set ``\\mathcal{S}``." 
const DOCS_NORM_CONE = "- `norm_cone=MOI.SecondOrderCone` - the cone related to the norm characterizing the proximal operator." 
const DOCS_ANALYTICAL_PROJ = "- `analytical_prox::Function=nothing` - the analytical form of the proximal operator for the given set. If provided, it replaces of `prox`." 

const DOCS_χ = "- `χ::Real` – The steps size value, corresponding to ``\\chi``."
const DOCS_χ_LARGE = "- `χ_large::Real=1e6` – A constant (arbitrarily) large value for the step size, corresponding to ``\\bar{\\chi}``." 
 
const DOCS_CK = "- `ck::Int` – The current counting parameter, corresponding to ``c_k``" 
const DOCS_CK1 = "- `ck1::Int` – The next counting parameter, corresponding to ``c_{k+1}``" 

const DOCS_α = "- `α::Real` – The auxiliary parameter, corresponding to ``\\alpha``." 
const DOCS_ϕ = "- `ϕ::Real` – The golden ratio step size, corresponding to ``\\phi``." 

const DOCS_TK = "- `tk::Real` - The current auxiliary coefficient, corresponding to ``\\theta_k``." 
const DOCS_TK1 = "- `xk1::AbstractVector` – The next auxiliary coefficient, corresponding to ``\\theta_{k+1}``."

const DOCS_X0 = "- `x0::AbstractVector` – The initial point, corresponding to ``\\mathbf{x}_0``."  
const DOCS_X1K = "- `x1k::AbstractVector` – The previous point, corresponding to ``\\mathbf{x}_{k-1}``." 
const DOCS_XK = "- `xk::AbstractVector` – The current point, corresponding to ``\\mathbf{x}_k``."  
const DOCS_XK1 = "- `xk1::AbstractVector` – The next point, corresponding to ``\\mathbf{x}_{k+1}``." 

const DOCS_Y0 = "- `y0::AbstractVector` – The initial auxiliary point, corresponding to ``\\mathbf{y}_0``."  
const DOCS_Y1K = "- `y1k::AbstractVector` – The previous point, corresponding to ``\\mathbf{y}_{k-1}``." 
const DOCS_YK = "- `yk::AbstractVector` – The current auxiliary point, corresponding to ``\\mathbf{y}_k``." 
const DOCS_YK1 = "- `yk1::AbstractVector` – The next auxiliary point, corresponding to ``\\mathbf{y}_{k+1}``."

const DOCS_S1K = "- `s1k::AbstractVector` – The previous point, corresponding to ``\\mathbf{s}_{k-1}``." 
const DOCS_SK = "- `sk::AbstractVector` – The current auxiliary point, corresponding to ``\\mathbf{s}_k``."

const DOCS_Z0 = "- `z0::AbstractVector` – The initial auxiliary point, corresponding to ``\\mathbf{z}_0``."
const DOCS_Z1K = "- `z1k::AbstractVector` – The previous point, corresponding to ``\\mathbf{z}_{k-1}``."
const DOCS_ZK = "- `zk::AbstractVector` – The current auxiliary point, corresponding to ``\\mathbf{z}_k``."
const DOCS_ZK1 = "- `zk1::AbstractVector` – The next auxiliary point, corresponding to ``\\mathbf{z}_{k+1}``."


