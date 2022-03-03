include("../src/OneRule.jl")

using Test
import StatsBase
using DataFrames
using .OneRule

### create test data 

weather = DataFrame(
    outlook = ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast", "sunny", "sunny", "rainy",  "sunny", "overcast", "overcast", "rainy"],
    temperature = ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot", "mild"],
    humidity = ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"],
    windy = ["false", "true", "false", "false", "false", "true", "true", "false", "false", "false", "true", "true", "false", "true"]
)

play = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"]

#### helper functions for checking trees and nodes

function check_oneTree(ot::OneTree, column::AbstractVector, target_labels)
    @test ot.inst_count == length(column)               # all instances used?

    # all target lables used?
    @test length(ot.target_lables) == length(target_labels)
    @test sum([t ∈ ot.target_lables for t in target_labels]) == length(target_labels)

    # check the nodes of the tree
    fvals = unique(column)
    fcounts = StatsBase.countmap(column)
    @test length(ot.nodes) == length(fvals) # all feature values used?
    for n in ot.nodes
        @test n.val_count == fcounts[n.feature_value]   # is the frequency of the feature value correct?
        @test n.feature_value ∈ fvals                   # is this a valid feature value?
        @test n.prediction ∈ target_labels              # is `prediction` a valid target label?
    end
end

### execute the tests

at = all_trees(weather, play)
t = get_best_tree(weather, play)
target_labels = unique(play)
@info("Test trees created for $(size(weather)[1]) instances and $(size(weather)[2]) features")

@test length(at) == size(weather)[2]                    # number of trees ok?
@info("Number of trees: $(length(at)) - correct")

# check all trees
@info("Testing each tree - start")
for col in 1:length(at)
    @info("    Testing tree: $(at[col].feature_name)")
    @test at[col].feature_name == names(weather)[col]   # feature names occur in training data?
    @test error_rate(t) ≤ error_rate(at[col])           # is `t` the tree with the smallest error rate?
    check_oneTree(at[col], weather[:, col], target_labels)
end
@info("Testing each tree - done")

@info("Predicting ...")
yhat = predict(t, weather)
@test length(play) - count(yhat .== play) == t.err_count # is the error count of the tree correct?
@info("Number of misclassified instances ($(t.err_count)) is correct")

