
# Load libraries
import pandas
import numpy
import json
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.externals import joblib

# Load dataset
url = "Dataset/adult.csv"
# url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
# names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
df = pandas.read_csv(url)

# filling missing values
col_names = df.columns
for c in col_names:
    df[c] = df[c].replace("?", numpy.NaN)

df = df.apply(lambda x:x.fillna(x.value_counts().index[0]))

df.replace(['Divorced', 'Married-AF-spouse', 
              'Married-civ-spouse', 'Married-spouse-absent', 
              'Never-married','Separated','Widowed'],
             ['divorced','married','married','married',
              'not married','not married','not married'], inplace = True)

category_col =['workclass', 'race', 'education','marital-status', 'occupation',
               'relationship', 'gender', 'native-country', 'income'] 
labelEncoder = preprocessing.LabelEncoder()

mapping_dict={}
for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col]=le_name_mapping
df=df.drop(['fnlwgt','educational-num'], axis=1)
df.head()

X = df.values[:, 0:12]
Y = df.values[:,12]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)


dt_clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)
### Desicion Tree with Information Gain ###

dt_clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=5, min_samples_leaf=5)

dt_clf_entropy.fit(X_train, y_train)

y_pred_gini = dt_clf_gini.predict(X_test)
y_pred_en = dt_clf_entropy.predict(X_test)

print ("Desicion Tree using Gini Index\nAccuracy is ", accuracy_score(y_test,y_pred_gini)*100 )
print ("Desicion Tree using Information Gain\nAccuracy is ", accuracy_score(y_test,y_pred_en)*100 )


##############################

#import pickle
import joblib

#creating and training a model
#serializing our model to a file called model.pkl
#pickle.dump(dt_clf_gini, open("NNmodel/model_yatin.pkl","wb"))
filename = 'model_yatin.sav'
joblib.dump(dt_clf_gini,filename)


