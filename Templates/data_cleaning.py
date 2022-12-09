"""
Template for typical Data Cleaning tasks
"""
import numpy as np

# Imputation
data = data.fillna(data.mean())
data = data.fillna(data.mode())
data.loc[(data.col > 100), 'col'] = 100 # generic replace

# dropping duplicates and nulls
data = data.drop_duplicates()
data = data.dropna()

# Clipping data

# by percentile
def percentile_clip(x, low_bound, high_bound):
    low_clip = x.quantile(low_bound)
    high_clip = x.quantile(high_bound)

    return x.clip(low_clip, high_clip)


# +/- 3 stdevs from mean
def three_std_clip(x):
    mean = np.mean(x)
    std = np.std(x)

    low_clip = mean - 3 * std
    high_clip = mean + 3 * std

    return x.clip(low_clip, high_clip)


# +/- 1.5 IQRs
def IQR_clip(x):
    q1 = x.quantile(.25)
    q3 = x.quantile(.75)

    IQR = q3 - q1
    low_clip = q1 - (1.5 * IQR)
    high_clip = q3 + (1.5 * IQR)

    return x.clip(low_clip, high_clip)