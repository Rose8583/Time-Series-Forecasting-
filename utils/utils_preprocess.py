###utils/data_preprocess.py
return float(np.sqrt(mean_squared_error(a, b)))

def mae(a, b):
return float(mean_absolute_error(a, b))

def mape(a, b):
a, b = np.array(a), np.array(b)
denom = np.where(np.abs(a) < 1e-8, 1e-8, a)
return float(np.mean(np.abs((a - b) / denom)) * 100)

###utils/cross_validation.py
#python
from typing import List, Tuple

def rolling_origin_splits(n_samples: int, initial_train: int, step: int, horizon: int, max_folds: int=10):
folds = []
start = initial_train
fold = 0
while start + horizon <= n_samples and fold < max_folds:
train_slice = slice(0, start)
test_slice = slice(start, start + horizon)
folds.append((train_slice, test_slice))
start += step
fold += 1
return folds
