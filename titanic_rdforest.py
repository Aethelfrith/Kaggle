import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report as cr
from sklearn import preprocessing as pp
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve


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

def preprocess_titanic_data(input_data,features,exists_y = True):
	input_data = input_data.copy()
	cols_x_keep = features

	#Select a subset of features to train on
	cols_y_keep = ["Survived"]
	
	X = input_data[cols_x_keep]
	if exists_y:
		y = input_data[cols_y_keep]

	#Replace nan values in the embarked column with a random choice of 'C','Q','S'
	X = fillna_w_rand_subset(X,"Embarked") 

	#Replace NaNs in the Age column with the average
	X = X.fillna(X.mean())
	
	if (exists_y == True):
		return X,y
	else:
		return X

def replace_string_with_numeric(X,columns_numerise):
	X = X.copy()
	#Replace selected column values with integers
	
	#Extract the unique values for each column
	for col in columns_numerise:
		unique_elements = X.loc[:,col].unique()
		n_unique_elements = len(unique_elements)
		index_list = list(range(0,n_unique_elements))
		dict_components = zip(unique_elements,index_list)
		curr_dict = {key:value for (key,value) in dict_components}
		X.replace({col:curr_dict},inplace = True)	
#	embarked_dict = {'C':0,'Q':1,'S':2}
#	X.replace({"Embarked":embarked_dict},inplace = True)
#	sex_dict = {'male':0,'female':1}
#	X.replace({'Sex':sex_dict},inplace = True)
	return X
	

def combine_features_titanic(X,train_feature_names=None):
	X = X.copy()
	X['FamSize'] = X['Parch'] + X['SibSp'] + 1
	X['isAlone'] = 1
	X.loc[X['FamSize']>1,'isAlone'] = 0
	if not train_feature_names == None:
		train_feature_names = train_feature_names.copy()
		train_feature_names.append('FamSize')
		train_feature_names.append('isAlone')
		return X, train_feature_names
	else:
		return X
		
def extract_and_add_titles(X,name_series,rare_name_thresh = 10):
	X = X.copy()
#	name_series = training_data['Name']
	given_names = name_series.str.split(', ',expand=True)[1]
	titles = given_names.str.split(".", expand=True)[0]
#	print("Unique titles:", titles.unique())
	name_counts = titles.value_counts()
	rare_names = name_counts.loc[name_counts.values < rare_name_thresh]
#	print(titles.values)
	rare_name_indices = [True if name in rare_names else False for name in titles.values]
	titles.loc[rare_name_indices] = 'Misc'
	X['Title'] = titles
#	print(X_train_val)
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
#	print(np_array_poly)
	polyfeature_names = poly_feature_factory.get_feature_names(df.columns)
#	print("Is multiindex: ",isinstance(polyfeature_names, pd.MultiIndex))
#	print(type(polyfeature_names))
	#Make a new dataframe
#	print(np_array_poly.shape)
	df = pd.DataFrame(np_array_poly, columns = polyfeature_names)
#	print(df)
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


training_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
#gender_submission = pd.read_csv('gender_submisson.csv')

#Explore the uncleaned data
#print(training_data.head())

#Extract the column names
#column_names = list(training_data.columns)
#print("Column names: \n",column_names)

#Count the number of survivors
#n_survivors = training_data["Survived"].value_counts()
#print("Survived (1), not survived (0): \n",n_survivors)
#The sample is not very skewed

#Select a set of features to train on
train_features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

#Clean the training data
X_train_val,y_train_val = preprocess_titanic_data(training_data, features = train_features)

#Manually add features
X_train_val, train_features_train = combine_features_titanic(X_train_val,train_features)

#Extract titles from names
name_series = training_data['Name']
X_train_val = extract_and_add_titles(X_train_val, name_series)

#Convert categorical data to numeric data
columns_numerise = ['Sex','Embarked','Title']
X_train_val = replace_string_with_numeric(X_train_val, columns_numerise)

#Clean the test data similarily
#Manually add features
X_test = preprocess_titanic_data(test_data, features = train_features, exists_y = False)
X_test, train_features_test = combine_features_titanic(X_test,train_features)

#Add polynomial features
poly_degree = 1
include_bias = False
interaction_only = False
X_train_val = polynomialize_df(X_train_val, poly_degree, include_bias, interaction_only)
train_features = X_train_val.columns.tolist() #Not sure why this is needed

#X_test = polynomialize_df(X_test, poly_degree, include_bias, interaction_only)


#Try normalising the data
#normalise_mode = 'std'
normalise_mode = 'None'
X_train_val = normalise_entire_dataframe(X_train_val, mode = normalise_mode)




##Explore the cleaned training data
##Plot where there are NaN elements
#spy(X_train_val)

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


#TRAINING
#Set global random seed
random_seed = 1234
#random_seed = 2017
#random_seed = 3972153975

#Subset training and validation data
X_train, X_val, y_train, y_val = train_test_split(X_train_val,y_train_val,random_state = random_seed)

#Run a machine learning algorithm
n_estimators = 100
estimator = RandomForestClassifier(n_estimators = n_estimators, random_state = random_seed)

estimator.fit(X_train,np.ravel(y_train))
y_pred_train = estimator.predict(X_val)
y_pred_train_self = estimator.predict(X_train)

#Inspect the importance of features
#Make a DataFrame of the feature importances and column names
feature_importances = np.array([estimator.feature_importances_,])#Need to convert to np.array
importance_df = pd.DataFrame(feature_importances, columns = [train_features])
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
training_curve_title = 'RandomForestClassifier'
train_val_split_folds = 5
train_sizes = np.linspace(0.04,1.0,20)

#plot_learning_curve(estimator, X_train_val, np.ravel(y_train_val), title = training_curve_title, cv=train_val_split_folds,train_sizes = train_sizes)
#plt.show()

#Plot a cross-validation curve
CV_curve_title = 'RandomForestClassifier'

#CV_param_name = 'n_estimators'
#CV_pararam_range = np.array([2, 10, 30, 50, 80, 100])

CV_param_name = 'max_leaf_nodes'
CV_param_range = np.array(np.linspace(2,50,5).astype(int))

#plot_validation_curve(estimator, X_train_val, np.ravel(y_train_val), CV_param_name, CV_param_range, title = CV_curve_title, xlabel = 'Parameter', ylabel = 'Score')
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

	
