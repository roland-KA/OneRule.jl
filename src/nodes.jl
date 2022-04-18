# a rule within a decision tree for a single feature value
mutable struct OneNode
    feature_value       # feature value (label) for which the rules predicts
    prediction          # predicted class for `feature_value`
    val_count  :: Int   # number of occurences of this feature value (in training data)
    err_count  :: Int   # number of wrongly classified instances (out of `val_count``)
end

# fraction [0 .. 1[ of how many instances are wrongly classified by this rule
error_rate(n::OneNode) = n.err_count // n.val_count

# make all nodes for feature `column`
function get_nodes(column::AbstractVector, target::AbstractVector, target_labels :: AbstractVector)
    fvals = unique(column)              # all feature values
    nodes = Dict{String, OneNode}()     # for the results
 
    # for each feature value we need a Dict of Dicts to count 
    # frequencies of each possible feature value/target label combination
    freq_table = Dict(f => Dict(cat => 0 for cat in target_labels) for f in fvals)

    # let's count the observations for each attribute value
    for i in eachindex(column)
        freq_table[column[i]][target[i]] += 1
    end

    # build the nodes of the tree (one for each feature value)
    for f in fvals
        maxfreq, cat = findmax(freq_table[f]) # find the most frequent category (and the number of it's occurences)
        sumfreq = sum(values(freq_table[f]))  # number or all occurences of `f``
        errfreq = sumfreq - maxfreq           # wrongly classified occurences of `f``
        nodes[string(f)] = OneNode(f, cat, sumfreq, errfreq)
    end
    return(nodes)
end

# print a 'OneNode' nicely formatted
function Base.show(io::IO, ::MIME"text/plain", node::OneNode) 
    print(io, "OneNode: $(node.feature_value) -> $(node.prediction) (Err $(node.err_count)/$(node.val_count))")
end

# use the same formatting in compact mode
function Base.show(io::IO, node::OneNode)
    print(io, "$(node.feature_value) -> $(node.prediction) ($(node.err_count)/$(node.val_count))")
end
 #   = show(io, "text/plain", node)
