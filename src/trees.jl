# a one-level decision tree (categorical data)
mutable struct OneTree
    feature_name                             # name of the feature for which the tree is created
    nodes         :: Dict{String, OneNode}   # the rules within the decision tree
    target_labels :: AbstractVector          # possible target labels which may be predicted by this OneTree
    inst_count    :: Int                     # number of instances (rows)
    err_count     :: Int                     # number of wrongly classified instances (out of `inst_count`)         
end

# total error rate of the tree; computed from the error rates of its rules
error_rate(t::OneTree) = t.err_count // t.inst_count

"""
    get_best_tree(X, y::AbstractVector) -> OneTree

Finds the one-level decision tree with the lowest error rate (a `OneTree`). 
`X` contains the features and `y` the corresponding target values.
"""
function get_best_tree(X, y::AbstractVector)
    trees = all_trees(X, y)
    return(trees[argmin(trees)])
end

# overload the < and == operators, so that `argmin` works
Base.:isless(t1::OneTree, t2::OneTree) = (error_rate(t1) < error_rate(t2))
Base.:isequal(t1::OneTree, t2::OneTree) = (error_rate(t1) == error_rate(t2))

# generate for each feature in `X` a decision tree 
function all_trees(X, y::AbstractVector)
    feature_cols  = Tables.columns(X)   # all features (as columns)
    target_labels = unique(y)           # possible target labels
    inst_count    = length(y)           # number of instances in `X`

    trees = Vector{OneTree}()   # for the results
  
    # make a tree for each feature
    for col in Tables.columnnames(feature_cols)
        nodes = get_nodes(Tables.getcolumn(feature_cols, col), y, target_labels)        
        errsum = sum(n -> n.err_count, values(nodes))    # sum up all `err_count`s for this feature
        push!(trees, OneTree(String(col), nodes, target_labels, inst_count, errsum))
    end
    return(trees)
end

# predict classes for new data using a trained `OneTree`
function predict(ot::OneTree, Xnew)
    preds       = Vector{typeof(ot.target_labels[1])}()                             # for the results
    feature_col = Tables.getcolumn(Tables.columns(Xnew), Symbol(ot.feature_name))   # the column used for predicing

    for fval in feature_col
        push!(preds, ot.nodes[fval].prediction)
    end
    return(preds)
end


# print a 'OneTree' nicely formatted
function Base.show(io::IO, ::MIME"text/plain", tree::OneTree) 
    println(io, "OneTree: $(tree.feature_name) (Err $(tree.err_count)/$(tree.inst_count))")
    for n in values(tree.nodes)
        println(io, "    $(n)")
    end
end
