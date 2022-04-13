using Documenter
using EquationLearning

push!(LOAD_PATH,"../src/")
makedocs(sitename="EquationLearning.jl Documentation",
         pages = [
            "Home" => "home.md",
            "VandenHeuvel et al. (2022)" => "paper.md",
            "Tutorial" => "tut.md",
            "Manual" => "ref.md",
            "Index" => "index.md"
         ],
         format = Documenter.HTML(prettyurls = false),
         modules = [EquationLearning]
)
# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/DanielVandH/EquationLearning.jl.git",
    devbranch = "main"
)