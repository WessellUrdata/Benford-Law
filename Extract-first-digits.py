import pickle as pkl
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt

filename = "output.pkl"
with open(filename, "rb") as f:
    t = pkl.load(f)

features = t["data"]

# Get first digit of each value
extract_first_digit = lambda x: int(str(x)[0])
for arrays in features:
    for i in range(len(arrays)):
        arrays[i] = extract_first_digit(arrays[i])

# Benford's Law for the first digit
digits = np.arange(1, 10)
benfords_law_distribution = np.log10(1 + 1 / digits)

# Get number of features
N = len(features[0])

# Count the occurrences of each first digit
first_digits_counts = [
    np.histogram(array, bins=np.arange(1, 11))[0] for array in features
]
ic(first_digits_counts)

# plot the distribution of first digits
for first_digits_count in first_digits_counts:
    plt.plot(
        digits, first_digits_count / N, color="blue", label="First digit distribution"
    )
    plt.plot(
        digits,
        benfords_law_distribution,
        color="red",
        label="Benford's Law distribution",
    )
    plt.xlabel("First digit")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
