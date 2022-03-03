# OneRule.jl

Implementation of the *1-Rule* data mining algorithm using the Julia programming language.

For more information about the algorithm see e.g.: https://datacadamia.com/data_mining/one_rule or have look
at 
>    Witten, Ian H., Eibe Frank, and Mark A. Hall. 
>    *Data Mining Practical Machine Learning Tools and Techniques* 
>    Third Edition. Morgan Kaufmann, 2017.

The implementation of this algorithm has started as an example for a university course in data science at [Baden-Württemberg Cooperative State University  Karlsruhe](https://www.karlsruhe.dhbw.de/en/general/about-us.html). 

It has beend extended to offer full functionality (e.g. for predicting) and it has been adapted to conform to common ML interfaces (in order to make it easily available for meta-packages like [MLJ.jl](https://alan-turing-institute.github.io/MLJ.jl/dev/)).

The model works on categorical data for the features as well as the target class for training (using the `get_best_tree` function.) You find an example on how to use it in the `examples` directory.

The core algorithm used to explain the inner workings of the OneRule model in the above-mentioned course, can still be found in the branch `teaching`.
