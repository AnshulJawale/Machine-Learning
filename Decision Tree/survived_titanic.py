from conf import join_csv, save_model
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np

csv_path = join_csv("titanic")

df = pd.read_csv(csv_path)

inputs = df.drop(["Survived", 'Unnamed: 12', "Cabin", "Embarked", "Ticket"], axis="columns")
target = df["Survived"]

inputs["Pclass"] = LabelEncoder().fit_transform(inputs["Pclass"])
inputs["Sex"] = LabelEncoder().fit_transform(inputs["Sex"])
inputs["Age"] = LabelEncoder().fit_transform(inputs["Age"])
inputs["Fare"] = LabelEncoder().fit_transform(inputs["Fare"])
inputs["Parch"] = LabelEncoder().fit_transform(inputs["Parch"])
inputs["Name"] = LabelEncoder().fit_transform(inputs["Name"])

X_train, X_test, y_train, y_test = train_test_split(inputs, target)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print(model.predict(X_test))
print(np.array(y_test))
print(model.score(X_test, y_test))

save_model(model, "survived_titanic")