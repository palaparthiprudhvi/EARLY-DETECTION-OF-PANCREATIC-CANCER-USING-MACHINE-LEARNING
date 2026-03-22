import numpy as np 
import pandas as pd 
from sklearn import metrics 
import matplotlib.pyplot as plt
df=pd.read_csv("pradataset.csv")
df.isnull().sum()
df=df.drop("sample_id",axis=1)
df["patient_cohort"].unique()
df["diagnosis"].unique()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
df['patient_cohort']= label_encoder.fit_transform(df['patient_cohort'])
df['patient_cohort']=df['patient_cohort'].astype(int)
df['patient_cohort']=df['patient_cohort'].astype(int)
df['sample_origin']= label_encoder.fit_transform(df['sample_origin'])
df['sample_origin']=df['sample_origin'].astype(int)
df['sex']= label_encoder.fit_transform(df['sex'])
df['sex']=df['sex'].astype(int)
df['diagnosis'].unique()
df['stage']=df['stage'].fillna('0')
df['stage'].unique()
df['stage']= label_encoder.fit_transform(df['stage’])
df['stage']=df['stage'].astype(int)
df['benign_sample_diagnosis'].unique()
df['benign_sample_diagnosis']=df['benign_sample_diagnosis'].fillna("null")
df['plasma_CA19_9']=df['plasma_CA19_9'].fillna(df['plasma_CA19_9'].mean())
df['REG1A']=df['REG1A'].fillna(df['REG1A'].mean())
df=df.drop("benign_sample_diagnosis",axis=1)
X=df.drop("diagnosis",axis=1)
y=df['diagnosis']
# Build a Dataframe with Correlation between Features
corr_matrix = X.corr()
# Take absolute values of correlated coefficients
corr_matrix = corr_matrix.abs().unstack()
corr_matrix = corr_matrix.sort_values(ascending=False)
corr_matrix = corr_matrix[corr_matrix >= 0.8]
corr_matrix = corr_matrix[corr_matrix < 1]
corr_matrix = pd.DataFrame(corr_matrix).reset_index()
corr_matrix.columns = ['feature1', 'feature2', 'Correlation']
corr_matrix.head()
# Import label encoder
from sklearn import preprocessing
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
y= label_encoder.fit_transform(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=60)
from sklearn.ensemble import ExtraTreesClassifier
Etc=ExtraTreesClassifier(n_estimators=100,max_depth=6,min_samples_split=2,min_weight_fraction_leaf =0.0,n_jobs=-1)
etc.fit(X_train, y_train)
print(etc.score(X_test, y_test)*100)
y_pred9 = etc.predict(X_test)
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred9, average='macro')
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100) 
clf.fit(X_train, y_train)
y_pred8 = clf.predict(X_test)
from sklearn import metrics 
print()
# using metrics module for accuracy calculation
print( metrics.accuracy_score(y_test, y_pred8)*100)
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred8, average='macro')
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_test_predict2=dtc.predict(X_test)
test_accuracy=(metrics.accuracy_score(y_test,y_test_predict2)*100)
test_accuracy
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svc= SVC()
svc.fit(X_train,y_train)
y_test_predict4=svc.predict(X_test)
test_accuracy=(accuracy_score(y_test,y_test_predict4)*100)
test_accuracy
df.head()

