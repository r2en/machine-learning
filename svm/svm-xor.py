from sklearn import svm

xor_data = [
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

data = []
label = []
for row in xor_data:
    p = row[0]
    q = row[1]
    r = row[2]
    data.append([p, q])
    label.append(r)

clf = svm.SVC()
clf.fit(data, label)

pre = clf.predict(data)
print("予測結果:", pre)

ok = 0; total = 0
for idx, answer in enumerate(label):
    p = pre[idx]
    if p == answer: ok += 1
    total += 1
print("正解率:", ok, "/", total, "=", ok/total)

import pdb; pdb.set_trace()