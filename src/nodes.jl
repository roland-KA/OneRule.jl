# a rule within a decision tree for a single attribute value
mutable struct OneNode
    atr_value :: String     # symbol name of an attribute value (categorical data)
    prediction :: String    # predicted class for `atr_value`
    err_rate :: Real        # fraction [0 .. 1[ of how many instances are wrongly classified by this rule
    err_count :: Integer    # number of wrongly classified instances
end

# make all nodes for attribute `column`
function get_nodes(column::AbstractVector, observations::AbstractVector, uniq_obs :: AbstractVector)
    vals = unique(column)                         # possible attribute values
    nodes = Vector{OneNode}(undef, length(vals))  # for the results
    
    freq_table = Dict{String, Dict{String, Int}}()    # Dict of Dicts for counting frequencies
 
    # for each attribute value we need a Dict to count frequencies of each possible observation
    for v in vals                                     
        atr_freq = Dict(cat => 0 for cat in uniq_obs)
        freq_table[v] = atr_freq
    end

    # let's count the observations for each attribute value
    for i in eachindex(column)
        freq_table[column[i]][observations[i]] += 1
    end

    # build the nodes of the tree (one for each attribute value)
    for i in eachindex(vals)
        maxfreq, cat = findmax(freq_table[vals[i]]) # find the most frequent category (and the number of it's occurences)
        sumfreq = sum(values(freq_table[vals[i]]))  # number or all occurences
        errfreq = sumfreq - maxfreq                 # wrongly classified occurences
        nodes[i] = OneNode(vals[i], cat, errfreq / sumfreq, errfreq)
    end
    return(nodes)
end

# print a 'OneNode' nicely formatted
function Base.show(io::IO, ::MIME"text/plain", node::OneNode) 
    print(io, "OneNode: $(node.atr_value) -> $(node.prediction) (Err $(node.err_count), $(node.err_rate))")
end

# use the same formatting in compact mode
Base.show(io::IO, node::OneNode) = show(io, "text/plain", node)
