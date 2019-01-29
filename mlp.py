import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report, confusion_matrix  

dataset = pd.read_csv("data/dataBuffer.csv")

# Split data into features (X) and targets (y)
X = dataset.iloc[:,4:]
y = dataset.iloc[:,:4]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30) 

# Scale data
scaler = StandardScaler() 
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Learn
mlp = MLPRegressor(hidden_layer_sizes=(7, 7, 7), max_iter=1000)
mlp.fit(X_train, y_train)

# Test
prediction = mlp.predict([X_test[0]])
#print(confusion_matrix(y_test,predictions)) 
#print(classification_report(y_test, predictions)) 
score = mlp.score(X_test, y_test)
print(score) 
print(prediction)