"""
Example from the book: 
    Witten, Ian H., Eibe Frank, and Mark A. Hall. 
    Data Mining Practical Machine Learning Tools and Techniques 
    Third Edition. Morgan Kaufmann, 2017.

Depending on weather information like outlook, temperature, humidity and windy a game is played (or not).

A call to `get_best_tree` with these indepentend variables and the dependent outcome (`play`) results in 
a one-level decision tree for further predictions.
"""

using OneRule
using DataFrames

weather = DataFrame(
    outlook = ["sunny", "sunny", "overcast", "rainy", "rainy", "rainy", "overcast", "sunny", "sunny", "rainy",  "sunny", "overcast", "overcast", "rainy"],
    temperature = ["hot", "hot", "hot", "mild", "cool", "cool", "cool", "mild", "cool", "mild", "mild", "mild", "hot", "mild"],
    humidity = ["high", "high", "high", "high", "normal", "normal", "normal", "high", "normal", "normal", "normal", "high", "normal", "high"],
    windy = ["false", "true", "false", "false", "false", "true", "true", "false", "false", "false", "true", "true", "false", "true"]
)

play = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no"]

t = get_best_tree(weather, play)

# create some random data for predicting
N = 10
newWeather = DataFrames(
    outlook = rand(["sunny", "overcast", "rainy"], N),
    temperature = rand(["hot", "mild", "cool"], N),
    humidity = rand(["high", "normal"], N),
    windy = rand(["false", "true"], N)
)

yhat = predict(t, newWeather)