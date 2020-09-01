import numpy as np
from libtlda.iw import ImportanceWeightedClassifier

X = np.random.randn(10, 2)
# y = np.random.randint(10,)
y = np.array([1,0,1,0,1,1,0,1,0,1])
Z = np.random.randn(10, 2)
ImportanceWeightedClassifier()
clf = ImportanceWeightedClassifier()
clf.fit(X, y, Z)
print(clf.predict_proba(Z))