import pandas as pd
from sklearn import svm, metrics, cross_validation
import random, re

csv = pd.read_csv('iris.csv')

data = csv[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
label = csv["Name"]

clf = svm.SVC()
scores = cross_validation.cross_val_score(clf, data, label, cv=5)

print("各正解率=", scores)
print("正解率=", scores.mean())

import pdb ; pdb.set_trace()