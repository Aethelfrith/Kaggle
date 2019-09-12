import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report as cr
from sklearn import preprocessing as pp
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve

####################################
#Classify which passengers of the Titanic survive and which ones die.
#Here, a support vector machine is used for classification.
#This script focusses on improving the preprocessing step of data.
#Part of the code is inspired by some other Kaggle users.


#Define functions
def fillna_w_rand_subset(df,colname):
	#Fill nans in a DataFrame column with a random selection of other elements
	df = df.copy() #To remove the copy warning...
	
	#Record which row indices in the reference column are nan
	i_nan = df[colname].isnull().values
	n_nan = sum(i_nan)

	i_nonnan = np.invert(i_nan)
	nonnan_subseries = df.loc[i_nonnan,colname]
	nonnan_val_distr_series = nonnan_subseries.value_counts()
	
	#Generate n_nan random samples of other data
	rand_values = get_random_values_from_distr(nonnan_val_distr_series, n_nan)
	
	#Assign the random values to NaN location in the DataFrame
	df.loc[i_nan,[colname]] = rand_values
	return df

def get_random_values_from_distr(distr_series, n_rands):
	elements = distr_series.index
	prob_values = distr_series.values/sum(distr_series.values)
	return np.random.choice(elements,n_rands,p = prob_values)
	
def spy(X):
	#Make a plot of all NaN values in a DataFrame X
	X_values = X.values
	n_y,n_x = np.shape(X_values)

	nan_loc = np.where(X.isnull().values)
	nan_loc_x = nan_loc[1]
	nan_loc_y = nan_loc[0]

	scatter_size = 20
	plt.scatter(nan_loc_x,nan_loc_y,scatter_size)
	plt.xlim(0, n_x-1)
	plt.ylim(0, n_y-1)
	plt.show()
	return None

def engineer_famsize_and_isalone(X):
	X = X.copy()
	X['FamSize'] = X['Parch'] + X['SibSp'] + 1
	X['isAlone'] = 1
	X.loc[X['FamSize']>1,'isAlone'] = 0
	return X
		
def extract_and_add_titles(X,rare_name_thresh = 10):
	X = X.copy()
	name_series = X["Name"]

	given_names = name_series.str.split(', ',expand=True)[1]
	titles = given_names.str.split(".", expand=True)[0]

	name_counts = titles.value_counts()
	rare_names = name_counts.loc[name_counts.values < rare_name_thresh]

	rare_name_indices = [True if name in rare_names else False for name in titles.values]
	titles.loc[rare_name_indices] = 'Misc'
	X['Title'] = titles

	return X
		
def fill_with_mode(X,features):
	X = X.copy()
	for feature in features:
		X[feature].fillna(X[feature].mode()[0],inplace=True)
	return X
	
def fill_with_mean(X,features):
	X = X.copy()
	for feature in features:
		X[feature].fillna(X[feature].mean(),inplace=True)
	return X
	
def fill_with_median(X,features):
	X = X.copy()
	for feature in features:
		X[feature].fillna(X[feature].median(),inplace=True)
	return X

def split_target_and_rest(X,target_feature_name):
	X = X.copy()
	
	target_feature = X[target_feature_name]
	remaining_features = X.drop("Survived",axis=1)
	
	return remaining_features,target_feature

def onehot(X,features):
	X = X.copy()
	
	# Make binary variables out of categorical 
	X = pd.get_dummies(X,columns = features)
	
	return X
		
def normalise_entire_dataframe(X,mode = 'minmax'):
	if not isinstance(mode, str):
		raise TypeError('mode is not a string')# check if lower can be applied
	X = X.copy()

	if mode.lower() == 'minmax':
		X = (X - X.mean())/(X.max() - X.min())
	elif mode.lower() == 'std':
		X = (X - X.mean())/X.std()
	elif mode.lower() == 'none':
		X = X
	else:
		raise ValueError('mode invalid, choose one of minmax or std')
		X = None
	return X

def polynomialize_df(df, poly_degree, include_bias, interaction_only = False):
	df = df.copy()
	poly_feature_factory = pp.PolynomialFeatures(poly_degree,include_bias = include_bias, interaction_only = interaction_only)
	np_array_poly = poly_feature_factory.fit_transform(df)
	polyfeature_names = poly_feature_factory.get_feature_names(df.columns)
	#Make a new dataframe
	df = pd.DataFrame(np_array_poly, columns = [polyfeature_names,])
	return df

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

#Load data
training_data = pd.read_csv("train.csv",index_col=0)
test_data = pd.read_csv("test.csv",index_col=0)
#gender_submission = pd.read_csv('gender_submisson.csv')

#BEGIN Preprocessing

#Select a set of features to train on. Implicitly ignore/remove the remaining ones.
train_features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
drop_features = ['Cabin','Ticket']

#Keep only the train features and survived
training_data.drop(drop_features,axis=1,inplace=True)
test_data.drop(drop_features,axis=1,inplace=True)

#Split the target (Survived) from other columns
#Name the dataframe train_val to signify that it includes both training and validation data
target_feature_name = ["Survived"]
X_train_val,y_train_val = split_target_and_rest(training_data,target_feature_name)
X_test = test_data.copy()

#Engineer familysize and isAlone features
X_train_val = engineer_famsize_and_isalone(X_train_val)
X_test = engineer_famsize_and_isalone(X_test)

#Extract titles from names and remove the names column
X_train_val = extract_and_add_titles(X_train_val)
X_train_val.drop(["Name"],axis=1,inplace=True)
X_test = extract_and_add_titles(X_test)
X_test.drop(["Name"],axis=1,inplace=True)


#Update the train feature list
train_features = list(X_train_val)


#Replace nans with the mode in categorical feature columns and mean in the rest
#Define categorical and noncategorical features
categorical_features = ["Embarked","Sex","Pclass","isAlone","Title"]
noncategorical_features = ["Age","SibSp","Parch","Fare"]

X_train_val = fill_with_median(X_train_val,noncategorical_features)
X_train_val = fill_with_mode(X_train_val,categorical_features)
X_test = fill_with_median(X_test,noncategorical_features)
X_test = fill_with_mode(X_test,categorical_features)


#Convert categorical features into oneshot labels, that is, a dummy variable for each category
X_train_val = onehot(X_train_val,categorical_features)
X_test = onehot(X_test,categorical_features)


#Add polynomial features
poly_degree = 1
include_bias = False
interaction_only = False
X_train_val = polynomialize_df(X_train_val, poly_degree, include_bias, interaction_only)
train_features = X_train_val.columns

X_test = polynomialize_df(X_test, poly_degree, include_bias, interaction_only)


#Try normalising the data
normalise_mode = 'std'
#normalise_mode = 'None'
X_train_val = normalise_entire_dataframe(X_train_val, mode = normalise_mode)
X_test = normalise_entire_dataframe(X_test, mode = normalise_mode)

#END Preprocessing

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
kernel_fun = 'linear'
#kernel_fun = 'rbf'

#Best for rbs kernel, no polynomial features
#gamma = 0.01
#reg_param_C = 10

#Best for radial kernel, polynomial features
#gamma = 0.01
#reg_param_C = 1

#Best for linear kernel, no polynomial features
#gamma = 1
#reg_param_C = 0.1

#Best for linear kernel, polynomial features
gamma = 1
reg_param_C = 0.1



max_iter = 20000
decision_function_shape = 'ovr'
estimator = SVC(C = reg_param_C, gamma = gamma, kernel = kernel_fun, decision_function_shape = decision_function_shape, random_state = random_seed, max_iter = max_iter)

estimator.fit(X_train,np.ravel(y_train))
y_pred_train = estimator.predict(X_val)
y_pred_train_self = estimator.predict(X_train)

#Inspect the importance of features
#Make a DataFrame of the feature importances and column names
importance_df = pd.DataFrame(estimator.coef_, columns = train_features)
print("Importance of features: ")
print(importance_df)

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
CV_curve_title = 'Support vector classifier'
CV_param_name = 'C'
#CV_param_name = 'gamma'
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
#y_pred_test = estimator.predict(X_test)

	
