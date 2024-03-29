import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report as cr

#from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})


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

	#Receive a subset of features to train on
	cols_x_keep = features
	cols_y_keep = ["Survived"]
	
	X = input_data[cols_x_keep]
	if exists_y:
		y = input_data[cols_y_keep]

	#Replace nan values in the embarked column with a random choice of 'C','Q','S'
	X = fillna_w_rand_subset(X,"Embarked") 

	#Replace the embarked column values with integers, do the same with sex
	embarked_dict = {'C':0,'Q':1,'S':2}
	X.replace({"Embarked":embarked_dict},inplace = True)
	sex_dict = {'male':0,'female':1}
	X.replace({'Sex':sex_dict},inplace = True)
	
	#Replace NaNs in the Age column with the average
	X = X.fillna(X.mean())
	
	if (exists_y == True):
		return X,y
	else:
		return X
		
def normalise_entire_dataframe(X,mode = 'minmax'):
	if not isinstance(mode, str):
		raise TypeError('mode is not a string')# check if lower can be applied
	X = X.copy()

	if mode.lower() == 'minmax':
		X = (X - X.mean())/(X.max() - X.min())
	elif mode.lower() == 'std':
		X = (X - X.mean())/X.std()
	else:
		raise ValueError('mode invalid, choose one of minmax or std')
		X = None
	return X
		
	

#END Define functions


training_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
#gender_submission = pd.read_csv('gender_submisson.csv')

#Explore the uncleaned data
#print(training_data.head())

#Extract the column names
column_names = list(training_data.columns)
#print("Column names: \n",column_names)

#Count the number of survivors
#n_survivors = training_data["Survived"].value_counts()
#print("Survived (1), not survived (0): \n",n_survivors)
#The sample is not very skewed

#Select features to train on
train_features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]

#Clean the training data
X_train_val,y_train_val = preprocess_titanic_data(training_data,features = train_features)

#Clean the test data similarily
X_test = preprocess_titanic_data(test_data,features = train_features,exists_y = False)

#Try normalising the data
normalise_mode = 'std'
X_train_val = normalise_entire_dataframe(X_train_val, mode = normalise_mode)
#X_test = normalise_entire_dataframe(X_test, mode = normalise_mode)



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

#Subset training and validation data
X_train, X_val, y_train, y_val = train_test_split(X_train_val,y_train_val,random_state = random_seed)

#Run a machine learning algorithm
n_max_leaf_nodes = 3
estimator = DecisionTreeClassifier(max_leaf_nodes = n_max_leaf_nodes,random_state = random_seed)

estimator.fit(X_train,y_train)
y_pred_train = estimator.predict(X_val)

#Inspect the importance of features
#Make a series of the feature importances and column names
#print(train_features)
#print(estimator.feature_importances_)
importance_df = pd.DataFrame([estimator.feature_importances_,], columns = train_features)
print(importance_df)

#VALIDATION

#Compare the predictions with the real values
class_names = ['Diseased','Survived']
class_rep_val = cr(y_val,y_pred_train,target_names = class_names)
print(class_rep_val)

#Generate predictions from the test set
y_pred_test = estimator.predict(X_test)

	
