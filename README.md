# ML Project I 
## Higgs Boson Challenge
#####Samuele Caneschi, Maximo Cravero, Milo Imbeni

To reproduce our result, at first make sure to have ```Numpy``` installed and to download properly the dataset 
```train.csv``` and ```test.csv```. Then, run the file ```run.py```.

The other files content, which has been use in this project to get to the final result, is explained in detail in the 
following document.

###```data_preprocessing.py``` 
Contains all the functions used to apply different pre-processing methods to the data before building the model. The 
functions implemented in this section are:
- ```set_nan```, ```convert_nan```: respectively to set all -999 values to nan and to convert them to the mean, median
or mode;
- ```remove_constant_columns```: removes the constant columns;
- ```standardize_data```: to standardize the dataset;
- ```balance_all```, ```balance_fromnans```: to balance the dataset, so that we have an input with the same numbers of 
signals and backgrounds;
- ```build_poly```: gives as output the polynomial expansion of the input dataset;
- ```split_data```, ```split_data_jet```: to split the dataset based on the jet number feature;
- ```batch_iter```, ```build_k_indices```: functions that create a k-fold division of the starting
 dataset to perform cross validation of hyperparameters;
- ```cross_channel_features```: creates an array containing the products between every couple of features;
- ```remove_outliers```: returns the dataset after removing the outliers datapoints;
- ```transform_data```: performs the polynomial expansion of the dataset and adds to it the columns containing products
between the features;
- ```preprocess_data```: contains all the pre-processing methods we wnat to apply to the dataset.
 
###```implementations.py``` 
Contains all the functions associated with the different methods we used to build models and the functions used to 
perform simple operations necessary to obtain coefficients used in the implementation of the methods. The functions 
in this section are:
- ```compute_mse```, ```compute_rmse```: compute mse and rmse;
- ```compute_gradient```, ```calculate_hessian```: compute gradient and hessian;
- ```sigmoid```: express the sigmoid function;
- ```compute_negative_log_likelihood_loss```, ```compute_negative_log_likelihood_gradient```: compute the negative
log likelihood;
- ```compute_accuracy```: calculates the accuracy of the prediction; 
- ```least_squares```, ```least_squares_GD```, ```least_squares_SGD```: contain the algorithms to respectively calculate 
the solutions with the least squares analytical method, with the gradient descent and with the statistic gradient descent;
- ```ridge_regression```: contains the algorithm to calculate the solution with ridge regression method;
- ```logistic_regression```, : contains the algorithms to calculate the solution with the logistic regression method, 
using gradient descent;
- ```penalized_logistic_regression```: returns the loss and gradient that will be used in the regularized method;  
- ```reg_logistic_regression_```: contains the algorithm to calculate the solution with the regularized logistic 
regression method, using gradient descent;
- ```cross_validation```: performs cross validation using two modes: ```default``` or ```jet_groups```, which is used 
when the dataset has been split based on the jet number. 

###```utilities.py``` 
Contains all the functions doing simple operations to calculate coefficients or indicators that are useful in other
implementations. The functions implemented in this section are:
- ```compute_f1score```, ```matthews_coeff```, ```calculate_recall_precision_accuracy```: calculate different metrics to 
evaluate the efficiency of the method used for prediction;
- ```create_confusion_matrix```: creates the confusion matrix, used to compute the different metrics;
- ```obtain_best_params```: .

###```proj1_helpers.py``` 
Contains the following functions:
- ```load_csv_data```: loads the dataset;
- ```predict_labels```: predicts the classification (signal/background) of the test dataset based on the model built;
- ```create_csv_submission```: creates a submission document to test the performance of the model built. 

### Other files
Other files, such as ```least_squares.py```, ```ridge_regression```, ```logistic_regression.py``` and 
```reg_logistic_regression```, are used to perform hyperparameters exploration and to understand which is the best 
combination of them for each method implemented. The file  ```least_squares.py``` has three different operating modes: 
```ls```, ```ls_GD``` and ```ls_SGD```, to distinguish the analytic method, from gradient and stochastic gradient descent.