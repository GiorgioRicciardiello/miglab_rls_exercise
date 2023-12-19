# Data Splits

### Train Set:
	Number of Observations: 195
	Number of Features: 44
	Count of Unique Values in Target:
		positive: 140
		negative: 55

## Test Set:
	Number of Observations: 49
	Number of Features: 44
	Count of Unique Values in Target:
		positive: 32
		negative: 17

# Confusion Matrix:

## Confusion Matrix (Train Set):
|          |   negative |   positive |
|----------|------------|------------|
| negative |         39 |         16 |
| positive |          2 |        138 |

## Confusion Matrix (Test Set):
|          |   negative |   positive |
|----------|------------|------------|
| negative |          5 |         12 |
| positive |          5 |         27 |

# Metrics Table:
|              |   precision |   recall |   f1-score |   support | split   | ObsCount   |
|--------------|-------------|----------|------------|-----------|---------|------------|
| negative     |       0.951 |    0.709 |      0.812 |    55     | train   | 55         |
| positive     |       0.896 |    0.986 |      0.939 |   140     | train   | 140        |
| accuracy     |       0.908 |    0.908 |      0.908 |     0.908 | train   | -          |
| macro avg    |       0.924 |    0.847 |      0.876 |   195     | train   | -          |
| weighted avg |       0.912 |    0.908 |      0.903 |   195     | train   | -          |
| negative     |       0.5   |    0.294 |      0.37  |    17     | test    | 17         |
| positive     |       0.692 |    0.844 |      0.761 |    32     | test    | 32         |
| accuracy     |       0.653 |    0.653 |      0.653 |     0.653 | test    | -          |
| macro avg    |       0.596 |    0.569 |      0.565 |    49     | test    | -          |
| weighted avg |       0.626 |    0.653 |      0.625 |    49     | test    | -          |

# Model Parameters:

	objective: multi:softmax
	num_class: 2
	n_estimators: 20
	eta: 0.01
	max_depth: 6
	reg_alpha: 2
	reg_lambda: 1.3
	gamma: 0.6
	subsample: 0.6
	colsample_bytree: 0.6
	device: cuda
