"""
Implementation of the 1-Rule-algorithm, which finds classification rules from a set of instances.
The result is a one-level deicsion tree (here of type `OneTree`). 
For more information about the 1-Rule-algorithm see e.g.: https://datacadamia.com/data_mining/one_rule

The implementation of this algorithm is intended for teaching purposes; it's not for production use.
"""

module OneRule

using DataFrames

export  OneTree,
        OneNode,
        get_best_tree

include("nodes.jl")
include("trees.jl")

end # module

