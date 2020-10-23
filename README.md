# ML Project 1
#####Samuele Caneschi, Maximo Cravero, Milo Imbeni

To reproduce our result, at first make sure to have ```Numpy``` installed and to download properly the dataset 
```train.csv``` and ```test.csv```. Then, run the file ```run.py```.

The other files content, which has been use in this project to get to the final result, is explained in detail in the 
following document.

###```data_exploration.py``` 
Contains all the functions used to understand the kind of data we have analyzed and their 
properties. Some example of the function you can find are:
- ```count_nan```, ```chech_nan_positions```: to count and find out the position of nan values;
- ```cut_dataset```: cuts some pre-selected features from the dataset;
- ```coviariance matrix```: the output is a matrix containing the values of the covariance between the different features;
- ```lin_dep```: the output contains all the couples of features that are linearly dependent;
- ```principal_component_analysis```: returns the explanatory power of each feature;
- ```set_cov_inf```, ```corr_col```: the output of the second function, that uses the first to do its operations, is a 
list that contains all the couples of features that has a correlation higher than a pre-selected threshold.

###```data_preprocessing.py``` 
Contains all the functions used to apply different pre-processing methods to the data,
that are useful to treat them before building the model. The functions implemented in this section are:
- ```set_nan```, ```convert_nan```: respectively to set all -999 values to nan and to convert them to the mean, median
or ...;
- ```standardize_data```: to standardize the dataset;
- ```balance_all```, ```balance_fromnans```: to balance the dataset, so that we have an input with the same numbers of 
signals and backgrounds;
- ```build_poly```: gives as output the polynomial expansion of the input dataset;
- ```split_data```, ```split_data_jet```: to split the dataset based on the jet number feature;
- ```batch_iter```, ```generate_batch```, ```build_k_indices```: functions that create a k-fold division of the starting
 dataset to perform cross validation of hyperparameters;
- ```preprocess_data```: contains all the pre-processing methods we wnat to apply to the dataset.

 
###```implementations.py``` 
Contains all the functions associated to the different methods we used to build models, from 
the pre-processed data. The functions implemented in this section are:
- ```least_squares```, ```least_squares_GD```, ```least_squares_SGD```: contain the algorithms to respectively calculate 
the solutions with the least squares analytical method, with the gradient descent and with the statistic gradient descent;
- ```ridge_regression```: contains the algorithm to calculate the solution with ridge regression method;
- ```logistic_regression_GD```, ```logistic_regression_SGD```: contain the algorithms to calculate the solution 
 with the logistic regression method, using respectively gradient descent and statistic gradient descent;
- ```reg_logistic_regression_```: returns the loss, gradient and hessian matrix that will be used in the regularized
methods' implementations;  
- ```reg_logistic_regression_GD```, ```logistic_regression_SGD```: contain the algorithms to calculate the solution with 
the logistic regression method, using respectively gradient descent and statistic gradient descent;
- ```cross_validation```: performs cross validation using two modes: ```default``` or ```jet_groups```, which is used 
when the dataset has been split based on the jet number. 

###```utilities.py``` 
Contains all the functions doing simple operations to calculate coefficients or indicators that are useful in other
implementations. The functions implemented in this section are:
- ```compute_mse```, ```compute_rmse```: compute mse and rmse;
- ```compute_gradient```, ```calculate_hessian```: compute gradient and hessian;
- ```sigmoid```: express the sigmoid function;
- ```compute_negative_log_likelihood_loss```, ```compute_negative_log_likelihood_gradient```: compute the negative
log likelihood;
- ```compute_accuracy```, ```compute_f1score```, ```matthews_coeff```: calculate different metrics to evaluate the 
efficiency of the method used for prediction;
- ```create_confusion_matrix```, ```calculate_recall_precision_accuracy```: respectively creates the confusion matrix 
and calculates the main metrics from the confusion matrix.

###```proj1_helpers.py``` 
Contains the following functions:
- ```load_csv_data```: loads the dataset;
- ```predict_labels```: predicts the classification (signal/background) of the test dataset based on the model built;
- ```create_csv_submission```: creates a submission document to test the performance of the model built. 

### Other files
Other files, such as ```least_squares.py```, ```logistic_regression.py```, ```reg_logistic_regression``` and
 ```ridge_regression```, are used to perform hyperparameters exploration and to understand which is the best combination
 of them for each method implemented. The file  ```least_squares.py``` has three different operating modes: ```ls```, 
 ```ls_GD``` and ```ls_SGD```, to distinguish the analytic method, from gradient and stochastic gradient descent.
 Similarly, the files  ```logistic_regression.py``` and ```reg_logistic_regression.py```have two different operating 
 modes: ```lr_GD```/```reg_lr_GD``` and ```lr_SGD```/```reg_lr_GD```, to distinguish gradient from stochastic gradient 
 descent methods. 