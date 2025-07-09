push!(LOAD_PATH,"../src/")

using Documenter, Monviso

makedocs(
    sitename="Monviso.jl",
    remotes=nothing,
    pages=[
        "Home" => "index.md",
        "API" => "api.md"
    ]
)
