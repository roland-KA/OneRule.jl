# using OneRule
using DataFrames

weather = DataFrame(
    outlook = ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast", "sunny", "sunny", "rainy",  "sunny", "overcast", "overcast", "rainy"],
    temperature = ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot", "mild"],
    humidity = ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"],
    windy = ["false", "true", "false", "false", "false", "true", "true", "false", "false", "false", "true", "true", "false", "true"]
)

play = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"]

OneRule.get_best_tree(weather, play)
