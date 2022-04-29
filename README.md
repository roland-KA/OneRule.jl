# OneRule.jl

Implementation of the *1-Rule* data mining algorithm by Robert Holte (*"Very simple classification rules perform well on most commonly used datasets"* in: Machine Learning 11.1 (1993), pp. 63-90) using the Julia programming language.

For more information about the algorithm see e.g.: [Machine Learning - (One|Simple) Rule](https://datacadamia.com/data_mining/one_rule), [OneRClassifier - One Rule for Classification](http://rasbt.github.io/mlxtend/user_guide/classifier/OneRClassifier/) or have a look at 
>    Witten, Ian H., Eibe Frank, and Mark A. Hall. 
>    *Data Mining Practical Machine Learning Tools and Techniques* 
>    Third Edition. Morgan Kaufmann, 2017, pp. 93-96

The implementation of this algorithm has started as an example for a university course in data science at [Baden-Württemberg Cooperative State University  Karlsruhe](https://www.karlsruhe.dhbw.de/en/general/about-us.html). 

It has beend extended to offer full functionality (e.g. for predicting) and it has been adapted to conform to common ML interfaces. It's now part of the meta-package [MLJ.jl](https://alan-turing-institute.github.io/MLJ.jl/dev/)).

The model works on categorical data for the features as well as the target class for training (using the `get_best_tree` function.) You find an example on how to use it in the `examples` directory.

For a description on its use within the context of the MLJ package, just type
```Julia
using MLJ
doc("OneRuleClassifier", pkg="OneRule")
```

The core algorithm used to explain the inner workings of the OneRule model in the above-mentioned course, can still be found in the branch `teaching`.
