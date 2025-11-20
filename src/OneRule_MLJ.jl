# MLJ interface for the OneRule model

import MLJModelInterface   
using CategoricalArrays    
const MMI = MLJModelInterface  # We need to repeat it here

export OneRuleClassifier, fit

mutable struct OneRuleClassifier <: MMI.Deterministic
end

function MMI.fit(model::OneRuleClassifier, verbosity, X, y)
    tree = get_best_tree(X, y)
    all_classes = MMI.classes(y[1])                         #   used in `predict` to return a prediction with the same levels
    fitresult = (tree, all_classes)      
    cache = nothing
    report = (                                              # We report ...
        tree = tree,                                        #   the OneTree
        nrules = length(tree.nodes),                        #   number of rules it contains
        error_rate = error_rate(tree),                      #   error rate (fraction of wrongly classified instances)
        error_count = tree.err_count,                       #   number of wrongly classified instances
        classes_seen = tree.target_labels,                  #   list of target classes observed in training
        features = collect(String.(Tables.columnnames(X)))) #   feature names used
    return(fitresult, cache, report)
end

function MMI.predict(model::OneRuleClassifier, fitresult, Xnew)
    yhat = OneRule.predict(fitresult[1], Xnew)
    return(categorical(unwrap.(yhat), levels = fitresult[2]))        # a CategoricalArray with same levels like the target vector
end

MMI.fitted_params(::OneRuleClassifier, fitresult) = (tree = fitresult[1], all_classes = fitresult[2])

MMI.metadata_pkg.(OneRuleClassifier,
      name       = "OneRule",
      uuid       = "90484964-6d6a-4979-af09-8657dbed84ff",        # see your Project.toml
      url        = "https://github.com/roland-KA/OneRule.jl",     # URL to your package repo
      julia      = true,                                          # is it written entirely in Julia?
      license    = "MIT",                                         # your package license
    )

MMI.metadata_model(OneRuleClassifier,
      input_scitype    = MMI.Table(MMI.Finite),
      target_scitype   = AbstractVector{<: MMI.Finite},
	    load_path        = "OneRule.OneRuleClassifier"
    )


"""
$(MMI.doc_header(OneRuleClassifier))

`OneRuleClassifier` implements the *OneRule* method for classification by Robert Holte 
(\"Very simple classification rules perform well on most commonly used datasets\" 
in: Machine Learning 11.1 (1993), pp. 63-90). 

For more information see:
- Witten, Ian H., Eibe Frank, and Mark A. Hall. 
  Data Mining Practical Machine Learning Tools and Techniques Third Edition. 
  Morgan Kaufmann, 2017, pp. 93-96.
- [Machine Learning - (One|Simple) Rule](https://datacadamia.com/data_mining/one_rule)
- [OneRClassifier - One Rule for Classification](http://rasbt.github.io/mlxtend/user_guide/classifier/OneRClassifier/)

# Training data

In MLJ or MLJBase, bind an instance `model` to data with

    `mach = machine(model, X, y)``

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns
  each have one of the following element scitypes: `Multiclass`,
  `OrderedFactor`, or `<:Finite`; check column scitypes with `schema(X)`
- `y`: is the target, which can be any `AbstractVector` whose element
  scitype is `OrderedFactor` or `Multiclass`; check the scitype
  with `scitype(y)`
Train the machine using `fit!(mach, rows=...)`.

# Hyper-parameters

This classifier has no hyper-parameters.

# Operations

- `predict(mach, Xnew)`: return (deterministic) predictions of the target given
  features `Xnew` having the same scitype as `X` above. 

# Fitted parameters

The fields of `fitted_params(mach)` are:
- `tree`: the tree (a `OneTree`) returned by the core OneTree.jl algorithm
- `all_classes`: all classes (i.e. levels) of the target (used also internally to transfer `levels`-information to `predict`)

# Report

The fields of `report(mach)` are:
- `tree`: The `OneTree` created based on the training data
- `nrules`: The number of rules `tree` contains
- `error_rate`: fraction of wrongly classified instances
- `error_count`: number of wrongly classified instances
- `classes_seen`: list of target classes actually observed in training
- `features`: the names of the features encountered in training
  
# Examples
```
using MLJ

ORClassifier = @load OneRuleClassifier pkg=OneRule

orc = ORClassifier()

outlook = ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast", "sunny", "sunny", "rainy",  "sunny", "overcast", "overcast", "rainy"]
temperature = ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot", "mild"]
humidity = ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"]
windy = ["false", "true", "false", "false", "false", "true", "true", "false", "false", "false", "true", "true", "false", "true"]

weather_data = (outlook = outlook, temperature = temperature, humidity = humidity, windy = windy)
play_data = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"]

weather = coerce(weather_data, Textual => Multiclass)
play = coerce(play, Multiclass)

mach = machine(orc, weather, play)
fit!(mach)

yhat = MLJ.predict(mach, weather)       # in a real context 'new' `weather` data would be used
one_tree = fitted_params(mach).tree
report(mach).error_rate
```
See also
[OneRule.jl](https://github.com/roland-KA/OneRule.jl).
"""
OneRuleClassifier