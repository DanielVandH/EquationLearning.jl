using Documenter
using EquationLearning

makedocs(
    sitename = "EquationLearning",
    format = Documenter.HTML(),
    modules = [EquationLearning],
    pages = [
        "Introduction" => "index.md",
        "Gaussian Processes" => "intro.md",
        "Equation Learning" => "eql.md",
        "Other" => "other.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/DanielVandH/EquationLearning.jl.git"
)
