import pandas as pd
import numpy as np

from sklearn import preprocessing as pp


####################################
#Output a file containing preprocessed titanic training and valiation, and test data.


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

#Write output data to file

y_train_val_df = pd.DataFrame(y_train_val,columns = ['Survived'])
train_val_filename = 'titanic_trainval.csv'
target_filename = 'titanic_target.csv'
test_filename = 'titanic_test.csv'

X_train_val.to_csv(train_val_filename,index=False,encoding='utf-8')
y_train_val_df.to_csv(target_filename,index=True,encoding='utf-8')
X_test.to_csv(test_filename,index=False,encoding='utf-8')
	
