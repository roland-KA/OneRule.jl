"""
Implementation of the 1-Rule-algorithm (by Robert Holte), 
which finds classification rules from a set of instances based on categorical data.

The result is a one-level decision tree (here of type `OneTree`). 

For more information about the 1-Rule-algorithm see e.g.: https://datacadamia.com/data_mining/one_rule
"""

module OneRule

import Tables

export  OneTree,
        OneNode,
        get_best_tree,
        all_trees,
        get_nodes,
        predict,
        error_rate

include("nodes.jl")
include("trees.jl")
include("OneRule_MLJ.jl")

end # module

