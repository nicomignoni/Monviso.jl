using Documenter, DocumenterCitations, Monviso

bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"))

makedocs(
    sitename="Monviso.jl",
    pages=[
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Examples" => [
            "Feasibility problem" => "examples/feasibility-problem.md",
            "Skew symmetric" => "examples/skew-symmetric.md",
            "Two-players zero-sum games" => "examples/zero-sum-game.md",
            "Linear-quadratic games" => "examples/linear-quadratic-game.md",
            "Markov decision process" => "examples/markov-process.md",
            "Logistic regression" => "examples/logistic-regression.md"
        ],
        "API" => "api.md",
        "References" => "references.md"
    ],
    format = Documenter.HTML(
        edit_link="master",
        assets=["assets/favicon.ico"]
    ),
    repo=Remotes.GitHub("nicomignoni", "Monviso.jl"),
    plugins=[bib]
)

deploydocs(
    repo = "github.com/nicomignoni/Monviso.jl.git",
)
