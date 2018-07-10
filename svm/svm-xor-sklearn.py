import pandas as pd
from sklearn import svm, metrics

xor_input = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

xor_df = pd.DataFrame(xor_input)
xor_data  = xor_df.ix[:,0:1] # データ
xor_label = xor_df.ix[:,2]   # ラベル

clf = svm.SVC()
clf.fit(xor_data, xor_label)
pre = clf.predict(xor_data)

ac_score = metrics.accuracy_score(xor_label, pre)
print("正解率=", ac_score)

import pdb; pdb.set_trace()