from sklearn.datasets import load_digits
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from conf import save_model
from sklearn import preprocessing
import time

digits = load_digits()

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.01)
x_train = preprocessing.scale(x_train)

number_recognition_model = LogisticRegression()
number_recognition_model.fit(x_train, y_train)

score = number_recognition_model.score(x_test, y_test)

num = 1592

plt.gray()
plt.matshow(digits.images[num])
plt.show()

print(f"The number guessed is : {number_recognition_model.predict(digits.data[[num]])[0]}")
print(f"The correct number is : {digits.target[num]}")

save_model(number_recognition_model, "number_recognition_model")