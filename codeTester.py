from sklearn.model_selection import StratifiedKFold
import numpy as np
X = np.array([[1, 2], [5, 4], [1, 2], [3, 4], [0, 1], [2, 3]])
y = np.array([0, 0, 1, 1, 1, 1])
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)

K = 2
folds = list(StratifiedKFold(n_splits=K, shuffle=True, random_state=16).split(X, y))

# for train_index, test_index in skf.split(X, y):
# 	print("TRAIN:", train_index, "TEST:", test_index)
# 	X_train, X_test = X[train_index], X[test_index]
# 	y_train, y_test = y[train_index], y[test_index]

for j, (train_idx, test_idx) in enumerate(folds):
	print('\n===================FOLD=',j)
	X_train_cv = X[train_idx]
	y_train_cv = y[train_idx]
	X_holdout = X[test_idx]
	y_holdout= y[test_idx]

	print (X_train_cv, "X_train_cv")
	print (y_train_cv, "y_train_cv")
	print (X_holdout, "X_val")
	print (y_holdout, "y_val")
