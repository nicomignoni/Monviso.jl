push!(LOAD_PATH,"../src/")

using Documenter, Monviso

makedocs(
    sitename="Monviso.jl",
    pages=[
        "Home" => "index.md",
        "API" => "api.md"
    ],
    format = Documenter.HTML(
        edit_link="master",
        assets=["assets/favicon.ico"]
    ),
    repo=Remotes.GitHub("nicomignoni", "Monviso.jl"),

)
