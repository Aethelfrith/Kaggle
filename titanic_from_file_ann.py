import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report as cr
from sklearn import preprocessing as pp
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

####################################
#Classify which passengers of the Titanic survive and which ones die.
#Here, an artificial neural network is used for classification.
#This script focusses on improving the preprocessing step of data.
#Part of the code is inspired by some other Kaggle users.
#Load already preprocessed training, validation and test data.


#BEGIN Define functions

def plot_learning_curve(estimator, X, y, title = None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
        
    Note: From https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_validation_curve(estimator, X, y, param_name, param_range, title=None, 		xlabel = 'Parameter', ylabel = 'Score', ylim = None, cv = 5):
	plt.figure()
	plt.title(title)
	if ylim is not None:
		plt.ylim(*ylim)
	#NOTE: Strange that cv has to be hardcoded....
	train_scores, validation_scores = validation_curve(estimator, X, y, param_name, param_range, cv = 5)
	train_scores_mean = np.mean(train_scores,axis=1)
	train_scores_std = np.std(train_scores,axis=1)
	validation_scores_mean = np.mean(validation_scores,axis=1)
	validation_scores_std = np.std(validation_scores,axis=1)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	line_width = 2
	plt.semilogx(param_range,train_scores_mean,label = 'Training score', color='darkorange',lw = line_width)
	plt.fill_between(param_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.1, color='darkorange',lw = line_width)
	plt.semilogx(param_range,validation_scores_mean,label = 'CV score', color='navy',lw = line_width)
	plt.fill_between(param_range, validation_scores_mean - validation_scores_std, validation_scores_mean + validation_scores_std, alpha = 0.1, color='navy',lw = line_width)
	plt.legend(loc='best')
	return plt

#END Define functions

#Load preprocessed data
trainingval_data = pd.read_csv("titanic_trainval.csv",index_col=0)
target_data = pd.read_csv("titanic_target.csv",index_col=0)
test_data = pd.read_csv("titanic_test.csv",index_col=0)

#Rename and deep copy
X_train_val = trainingval_data.copy()
y_train_val = target_data.copy()
X_test = test_data.copy()

#Extract column names
train_features = trainingval_data.columns.values.tolist()

#BEGIN Inspection

##Check for linear correlations in the data
#X_train_val.plot(x='Sex', y='Age', style='o')
#X_train_val.plot(x='Age', y='Fare', style='o')
#plt.show()
#print(X_train_val.describe())

##Check for correlations
#X_corr = X_train_val.corr()
#sns.heatmap(X_corr,annot=True,cmap = plt.cm.Reds)
#plt.autoscale()
#plt.show()

##Plot the top elements
#print(X_train_val.head(10))

#END Inspection

#TRAINING
#Set global random seed
random_seed = 1234
#random_seed = 2017
#random_seed = 3972153975

#Subset training and validation data
X_train, X_val, y_train, y_val = train_test_split(X_train_val,y_train_val,random_state = random_seed)

#Run a machine learning algorithm
#Set parameters
#alpha = 0.0001
#hidden_layer_sizes = (10,)
#solver = 'lbfgs'
#activation = 'relu'
#max_iter = 200

alpha = 1
hidden_layer_sizes = (10,)
solver = 'lbfgs'
activation = 'relu'
max_iter = 200
verbose = False


estimator = MLPClassifier(alpha = alpha, hidden_layer_sizes = hidden_layer_sizes, solver = solver, activation = activation, random_state = random_seed, max_iter = max_iter,verbose = verbose)

estimator.fit(X_train,np.ravel(y_train))
y_pred_train = estimator.predict(X_val)
y_pred_train_self = estimator.predict(X_train)

#Inspect the importance of features
#Make a DataFrame of the feature importances and column names
#importance_df = pd.DataFrame(estimator.coef_, columns = train_features)
#print("Importance of features: ")
#print(importance_df)

#bias_term = estimator.intercept_
#print("Intercept: \n",bias_term)

#Plot the importances in a bar plot
#importance_df.plot(kind='bar',figsize = (10,6))
#plt.legend(ncol=3)
#plt.show()


#Plot a few dimensions of the data
#plot_column_1 = 'Sex'
#plot_column_2 = 'Pclass'
#plot_column_3 = 'SibSp'
#plot_classes = y_train
#plt_alpha = 0.2
#plt.scatter(x=X_train[plot_column_1].values,y=X_train[plot_column_2].values,alpha = plt_alpha, s=100*X_train[plot_column_3].values, c=plot_classes.values, cmap = 'viridis')
#plt.xlabel(plot_column_1)
#plt.ylabel(plot_column_2)
#plt.show()


#Plot a training curve
training_curve_title = 'Support vector classifier'
train_val_split_folds = 5
train_sizes = np.linspace(0.04,1.0,20)

#plot_learning_curve(estimator, X_train_val, np.ravel(y_train_val), title = training_curve_title, cv=train_val_split_folds,train_sizes = train_sizes)
#plt.show()

#Plot a cross-validation curve
CV_curve_title = 'MLP classifier'
#CV_param_name = 'hidden_layer_sizes'
CV_param_name = 'alpha'
CV_pararam_range = np.array([0.001,0.01,0.1,1,10,100])

#plot_validation_curve(estimator, X_train_val, np.ravel(y_train_val), CV_param_name, CV_pararam_range, title = CV_curve_title, xlabel = 'Parameter', ylabel = 'Score')
#plt.show()

#VALIDATION
#Display the error metrics on the training data
class_names = ['Diseased','Survived']
class_rep_train = cr(y_train,y_pred_train_self,target_names = class_names)
print("Performance on training data:")
print(class_rep_train)



#Compare the predictions with the real values
class_rep_val = cr(y_val,y_pred_train,target_names = class_names)
print("Performance on validation data:")
print(class_rep_val)

#Generate predictions from the test set
y_pred_test = estimator.predict(X_test)

#Write the predictions to file
temp = pd.DataFrame(np.arange(892,892+len(y_pred_test)),columns=['PassengerID'])
temp['Survived'] = y_pred_test
temp.to_csv("./submission.csv",header=True,index=False)
	
