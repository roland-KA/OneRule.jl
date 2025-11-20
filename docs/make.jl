using Documenter, OneRule

makedocs(
    modules = [OneRule],
    sitename = "OneRule", 
    format = Documenter.HTML(prettyurls = false),
    authors = "Roland Schätzle"
)
