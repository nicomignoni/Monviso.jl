using Documenter, DocumenterCitations, Monviso

bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"))

makedocs(
    sitename="Monviso.jl",
    pages=[
        "Home" => "index.md",
        "Getting started" => [
            "What's a VI?" => "getting-started/whats-a-vi.md",
            "A basic example" => "getting-started/basic-example.md"
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
