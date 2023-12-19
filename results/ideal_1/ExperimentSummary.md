# Data Splits

### Train Set:
	Number of Observations: 256
	Number of Features: 44
	Count of Unique Values in Target:
		positive: 140
		negative: 59
		non-responder: 57

## Test Set:
	Number of Observations: 64
	Number of Features: 44
	Count of Unique Values in Target:
		positive: 32
		non-responder: 19
		negative: 13

# Confusion Matrix:

## Confusion Matrix (Train Set):
|               |   negative |   positive |   non-responder |
|---------------|------------|------------|-----------------|
| negative      |         46 |         11 |               2 |
| positive      |          0 |        139 |               1 |
| non-responder |          1 |         10 |              46 |

## Confusion Matrix (Test Set):
|               |   negative |   positive |   non-responder |
|---------------|------------|------------|-----------------|
| negative      |          2 |          8 |               3 |
| positive      |          3 |         24 |               5 |
| non-responder |          2 |         14 |               3 |

# Metrics Table:
|               |   precision |   recall |   f1-score |   support | split   | ObsCount   |
|---------------|-------------|----------|------------|-----------|---------|------------|
| negative      |       0.979 |    0.78  |      0.868 |    59     | train   | 59         |
| positive      |       0.869 |    0.993 |      0.927 |   140     | train   | 140        |
| non-responder |       0.939 |    0.807 |      0.868 |    57     | train   | 57         |
| accuracy      |       0.902 |    0.902 |      0.902 |     0.902 | train   | -          |
| macro avg     |       0.929 |    0.86  |      0.888 |   256     | train   | -          |
| weighted avg  |       0.91  |    0.902 |      0.9   |   256     | train   | -          |
| negative      |       0.286 |    0.154 |      0.2   |    13     | test    | 13         |
| positive      |       0.522 |    0.75  |      0.615 |    32     | test    | 32         |
| non-responder |       0.273 |    0.158 |      0.2   |    19     | test    | 19         |
| accuracy      |       0.453 |    0.453 |      0.453 |     0.453 | test    | -          |
| macro avg     |       0.36  |    0.354 |      0.338 |    64     | test    | -          |
| weighted avg  |       0.4   |    0.453 |      0.408 |    64     | test    | -          |

# Model Parameters:

	objective: multi:softmax
	num_class: 3
	n_estimators: 20
	eta: 0.01
	max_depth: 6
	reg_alpha: 2
	reg_lambda: 1.3
	gamma: 0.6
	subsample: 0.6
	colsample_bytree: 0.6
	device: cuda
