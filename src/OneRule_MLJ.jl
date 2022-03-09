# MLJ interface for the OneRule model

import MLJModelInterface   
using CategoricalArrays    
const MMI = MLJModelInterface  # We need to repeat it here

export OneRuleClassifier, fit

mutable struct OneRuleClassifier <: MMI.Deterministic
end

function MMI.fit(model::OneRuleClassifier, verbosity, X, y)
    tree = get_best_tree(X, y)
    a_target_element = y[1]                             # for producing predictions of the same type
    fitresult = (tree, a_target_element,                
        tree.target_labels, names(X))                   # class names and feature names used
    cache = nothing
    report = (                                          # We report ...
        tree = tree,                                    #   the OneTree
        nrules = length(tree.nodes),                    #   number of rules it contains
        error_rate = error_rate(tree),                  #   error rate (percentage of wrongly classified instances)
        error_count = tree.err_count)                   #   number of wrongly classified instances
    return(fitresult, cache, report)
end

function MMI.predict(model::OneRuleClassifier, fitresult, Xnew)
    yhat = OneRule.predict(fitresult[1], Xnew)
    return(categorical(yhat, levels = levels(fitresult[2])))    # a CategoricalArray with same levels like the target vector
end

MMI.fitted_params(::OneRuleClassifier, fitresult) = (tree = fitresult[1], features = fitresult[4])

MMI.metadata_pkg.(OneRuleClassifier,
        name       = "OneTree",
        uuid       = "90484964-6d6a-4979-af09-8657dbed84ff",        # see your Project.toml
        url        = "https://github.com/roland-KA/OneRule.jl",     # URL to your package repo
        julia      = true,                                          # is it written entirely in Julia?
        license    = "MIT",                                         # your package license
        is_wrapper = false,                                         # does it wrap around some other package?
    )

MMI.metadata_model(OneRuleClassifier,
    input_scitype    = MMI.Table(MMI.Finite),
    target_scitype   = AbstractVector{<: MMI.Finite},
    supports_weights = false,
    descr            = "A simple OneRule mdodel for classification of categorical data.",
	load_path        = "OneRuleClassifier"
    )