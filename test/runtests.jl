include("../src/OneRule.jl")

using Test
import StatsBase
import Tables
using .OneRule
using MLJBase
using CategoricalArrays

### create test data 

outlook = ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast", "sunny", "sunny", "rainy",  "sunny", "overcast", "overcast", "rainy"]
temperature = ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot", "mild"]
humidity = ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"]
windy = ["false", "true", "false", "false", "false", "true", "true", "false", "false", "false", "true", "true", "false", "true"]

weather = (outlook = outlook, temperature = temperature, humidity = humidity, windy = windy)

play = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"]

#### helper function for checking trees and nodes

function check_oneTree(ot::OneTree, column::AbstractVector, target_labels)
    @test ot.inst_count == length(column)               # all instances used?

    # all target labels used?
    @test length(ot.target_labels) == length(target_labels)
    @test sum([t ∈ ot.target_labels for t in target_labels]) == length(target_labels)

    # check the nodes of the tree
    fvals = unique(column)
    fcounts = StatsBase.countmap(column)
    @test length(ot.nodes) == length(fvals) # all feature values used?
    for n in values(ot.nodes)
        @test n.val_count == fcounts[n.feature_value]   # is the frequency of the feature value correct?
        @test n.feature_value ∈ fvals                   # is this a valid feature value?
        @test n.prediction ∈ target_labels              # is `prediction` a valid target label?
    end
end

### execute the tests

at = all_trees(weather, play)
t = get_best_tree(weather, play)
target_labels = unique(play)
@info("Test trees created for $(length(weather)) instances and $(length(weather[1])) features")

@testset "Base" begin
    @test length(at) == length(weather)                         # number of trees ok?
    @info("Number of trees: $(length(at)) - correct")

    # check all trees
    @info("Testing each tree - start")
    for col in 1:length(at)
        col_name = Tables.columnnames(weather)[col]             # column name (as Symbol)
        @info("    Testing tree: $(at[col].feature_name)")
        @test at[col].feature_name == String(col_name)          # feature names occur in training data?
        @test error_rate(t) ≤ error_rate(at[col])               # is `t` the tree with the smallest error rate?
        check_oneTree(at[col], Tables.getcolumn(weather, col_name), target_labels)
    end
    @info("Testing each tree - done")

    @info("Predicting ...")
    yhat = OneRule.predict(t, weather)
    @test length(play) - count(yhat .== play) == t.err_count # is the error count of the tree correct?
    @info("Number of misclassified instances ($(t.err_count)) is correct")
end

@testset "MLJ interface" begin 
    @info("Testing MLJ interface - start")
    # create/adapt test data
    weather2 = coerce(weather, Textual => Multiclass)
    play_cat = coerce(play, Multiclass)

    @info("  MLJ: create model and machine")
    orc = OneRuleClassifier()
    mach = machine(orc, weather2, play_cat)

    @info("  MLJ: fit & predict")
    fit!(mach)
    yhat = OneRule.predict(t, weather)
    yhat_cat = MLJBase.predict(mach, weather2)
    t2 = report(mach).tree

    @info("  MLJ: levels of predictions match?")
    @test levels(play_cat) == levels(yhat_cat)              # same range of values?

    @info("  MLJ: compare trees")
    @test t.feature_name  == t2.feature_name
    @test t.err_count     == t2.err_count
    @test t.inst_count    == t2.inst_count
    @test t.target_labels == t2.target_labels
    for (n1, n2) in zip(values(t.nodes), values(t2.nodes))
        @info("  MLJ: comparing node $(n1.feature_value)")
        @test n1.feature_value == n2.feature_value
        @test n1.prediction    == n2.prediction
        @test n1.val_count     == n2.val_count
        @test n1.err_count     == n2.err_count
    end

    @info("  MLJ: compare predictions")
    @test yhat == yhat_cat

    @info("  MLJ: check `report` & `fitted_params`")
    fp = fitted_params(mach)
    r = report(mach)
    @test fp.tree === r.tree
    @test length(fp.all_classes) ≥ length(r.classes_seen)
    @test length(r) == 6

    @info("Testing MLJ interface - done")
end;