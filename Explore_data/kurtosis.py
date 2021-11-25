"""
Kurtosis is a measure of the "tailedness" of the probability distribution.

=> Like push or pull in vertical direction

Kurtosis > 3 => Leptokurtic
Kurtosis = 3 => Mesokurtic
Kurtosis < 3 => Platykurtic

Excess Kurtosis > 0 => Leptokurtic
Excess Kurtosis = 0 => Mesokurtic
Excess Kurtosis < 0 => Platykurtic

How to calculate Kurtosis ?

The measure of kurtosis is calculated as the fourth standardized moment of a distribution.
"""
from scipy.stats import kurtosis

x = [55, 78, 65, 98, 97, 60, 67, 65, 83, 65]

# Kurtosis
print(f"Kurtosis: {kurtosis(x, fisher=False)}")

# Excess Kurtosis
print(f"Excess Kurtosis: {kurtosis(x, fisher=True)}")