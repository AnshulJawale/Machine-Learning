import pandas as pd 
from sklearn import linear_model
from conf import join_csv, save_model
import word2number.w2n as w2n

filepath = join_csv("Candidates")
df = pd.read_csv(filepath)
df.rename(columns={"experience":"Experience", "test_score(out of 10)":"Test Score", "interview_score(out of 10)":"Interview Score",  "salary($)":"Salary (US$)"}, inplace=True)

df["Experience"] = df["Experience"].fillna("two")
df["Test Score"] = df["Test Score"].fillna(df["Test Score"].median())

experience = [w2n.word_to_num(i) for i in df["Experience"]]
df["Experience"] = experience

salary_model = linear_model.LinearRegression()
salary_model.fit(df[["Experience", "Test Score", "Interview Score"]], df["Salary (US$)"])

experience = int(input("Enter your experience : "))
test_score = int(input("Enter your test score : "))
interview_score = int(input("Enter your interview score : "))

if (10 >= test_score >= 4) and (10 >= interview_score >= 4) and (1 <= experience <= 50):
    print(f"Your Salary would be around $ {int(round(salary_model.predict([[experience, test_score, interview_score]])[0]))}")
else:
    print("You are not eligible for the job.")

save_model(salary_model, "salary_model")