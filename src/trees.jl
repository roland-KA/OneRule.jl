# a one-level decision tree
mutable struct OneTree
    atr_name :: String                  # name of the attribute for which the tree is created
    nodes :: AbstractVector{OneNode}    # the rules within the decision tree
    err_rate :: Real                    # total error rate of the tree; computed from the error rates of its rules
end

"""
    get_best_tree(train_data::AbstractDataFrame, observations::AbstractVector) -> OneTree

Finds the one-level decision tree with the lowest error rate (a `OneTree`). 
`train_data` contains the predictor attributes and `observations` the corresponding
observations.
"""
function get_best_tree(train_data::AbstractDataFrame, observations::AbstractVector)
    trees = all_trees(train_data, observations)
    return(trees[argmin(trees)])
end

# overload the < and == operators, so that `argmin` works
Base.:isless(t1::OneTree, t2::OneTree) = (t1.err_rate < t2.err_rate)
Base.:isequal(t1::OneTree, t2::OneTree) = (t1.err_rate == t2.err_rate)

# generate for each attribute in `train_data` a decision tree 
function all_trees(train_data::AbstractDataFrame, observations::AbstractVector)
    atr_names = names(train_data)       # all attribute names
    uniq_obs = unique(observations)     # possible observations
    inst_count = size(observations)[1]  # number of instances in `train_data`

    trees = Vector{OneTree}(undef, size(atr_names)) # for the results
  
    # make a tree for each attribute
    for i in eachindex(atr_names)
        nodes = get_nodes(train_data[:, atr_names[i]], observations, uniq_obs)
        errsum = 0          # sum up all `err_count`s for this attribute
        for n in nodes
            errsum += n.err_count
        end
        trees[i] = OneTree(atr_names[i], nodes, errsum / inst_count)
    end
    return(trees)
end