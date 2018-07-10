import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation, metrics

mr = pd.read_csv("mushroom.csv", header=None)

label = []
data = []
attr_list = []
for row_index, row in mr.iterrows():
    label.append(row.ix[0])
    row_data = []
    for v in row.ix[1:]:
        row_data.append(ord(v))
    data.append(row_data)

data_train, data_test, label_train, label_test = \
    cross_validation.train_test_split(data, label)

clf = RandomForestClassifier()
clf.fit(data_train, label_train)

predict = clf.predict(data_test)

ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print("正解率=", ac_score)
print("レポート=\n", cl_report)

import pdb ; pdb.set_trace()