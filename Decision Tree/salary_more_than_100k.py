from conf import save_model, join_csv
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

csv_path = join_csv("salaries")

df = pd.read_csv(csv_path)

#? company, job, degree, salary_more_then_100k

inputs = df.drop(["salary_more_then_100k"], axis="columns")
target = df["salary_more_then_100k"]

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

inputs["company"] = le_company.fit_transform(inputs["company"])
inputs["job"] = le_company.fit_transform(inputs["job"])
inputs["degree"] = le_company.fit_transform(inputs["degree"])

model = tree.DecisionTreeClassifier()
model.fit(inputs, target)
save_model(model, "salary_more_then_100k")

print(model.predict([[2, 1, 1]]))