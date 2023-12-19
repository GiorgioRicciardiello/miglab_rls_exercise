# Data Splits

### Train Set:
	Number of Observations: 420
	Number of Features: 44
	Count of Unique Values in Target:
		positive_both_negative: 356
		non-responder: 64

## Test Set:
	Number of Observations: 106
	Number of Features: 44
	Count of Unique Values in Target:
		positive_both_negative: 94
		non-responder: 12

# Confusion Matrix:

## Confusion Matrix (Train Set):
|                        |   non-responder |   positive_both_negative |
|------------------------|-----------------|--------------------------|
| non-responder          |              34 |                       30 |
| positive_both_negative |               0 |                      356 |

## Confusion Matrix (Test Set):
|                        |   non-responder |   positive_both_negative |
|------------------------|-----------------|--------------------------|
| non-responder          |               0 |                       12 |
| positive_both_negative |               0 |                       94 |

# Metrics Table:
|                        |   precision |   recall |   f1-score |   support | split   | ObsCount   |
|------------------------|-------------|----------|------------|-----------|---------|------------|
| non-responder          |       1     |    0.531 |      0.694 |    64     | train   | 64         |
| positive_both_negative |       0.922 |    1     |      0.96  |   356     | train   | 356        |
| accuracy               |       0.929 |    0.929 |      0.929 |     0.929 | train   | -          |
| macro avg              |       0.961 |    0.766 |      0.827 |   420     | train   | -          |
| weighted avg           |       0.934 |    0.929 |      0.919 |   420     | train   | -          |
| non-responder          |       0     |    0     |      0     |    12     | test    | 12         |
| positive_both_negative |       0.887 |    1     |      0.94  |    94     | test    | 94         |
| accuracy               |       0.887 |    0.887 |      0.887 |     0.887 | test    | -          |
| macro avg              |       0.443 |    0.5   |      0.47  |   106     | test    | -          |
| weighted avg           |       0.786 |    0.887 |      0.834 |   106     | test    | -          |

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
