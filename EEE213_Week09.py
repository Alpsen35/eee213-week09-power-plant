
import pandas as pd
import numpy as np
import sys

# 1. Load dataset
data = pd.read_csv("assets__data__eee213__ccpp.csv")

print("Data shape:", data.shape)

# 2. Separate features and target
X = data[["AT", "V", "AP", "RH"]]
y = data["PE"]

# 3. Split training and test data
X_train = X.iloc[:7654]
y_train = y.iloc[:7654]

X_test = X.iloc[7654:]
y_test = y.iloc[7654:]

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

# 4. Data quality checks
print("\n--- DATA QUALITY CHECKS ---")

print("Missing values:")
print(data.isnull().sum())
print("Total missing:", data.isnull().sum().sum())

print("Exact duplicate rows:", data.duplicated().sum())

print("\nMinimum values:")
print(data.min())

print("\nMaximum values:")
print(data.max())

# 5. Summary statistics
print("\n--- SUMMARY STATISTICS ---")

for column in ["AT", "PE"]:
    print("\n", column)
    print("Mean:", round(data[column].mean(), 3))
    print("Median:", round(data[column].median(), 3))
    print("Standard deviation:", round(data[column].std(), 3))

# 6. Correlation with PE
print("\n--- CORRELATION WITH PE ---")

correlations = data[["AT", "V", "AP", "RH", "PE"]].corr()["PE"]
print(correlations.drop("PE").round(3))

# 7. Baseline model
training_mean = y_train.mean()

baseline_predictions = np.full(len(y_test), training_mean)
baseline_mae = np.mean(np.abs(y_test - baseline_predictions))

print("\n--- SECTION 5 RESULTS ---")
print("Training mean:", round(training_mean, 3), "MW")
print("Baseline test MAE:", round(baseline_mae, 3), "MW")

# 8. Linear regression using AT
at_slope, at_intercept = np.polyfit(X_train["AT"], y_train, 1)

at_predictions = at_slope * X_test["AT"] + at_intercept
at_mae = np.mean(np.abs(y_test - at_predictions))

print("\nAT slope:", round(at_slope, 6))
print("AT intercept:", round(at_intercept, 6))
print("AT test MAE:", round(at_mae, 3), "MW")

# 9. Linear regression using V
v_slope, v_intercept = np.polyfit(X_train["V"], y_train, 1)

v_predictions = v_slope * X_test["V"] + v_intercept
v_mae = np.mean(np.abs(y_test - v_predictions))

print("\nV slope:", round(v_slope, 6))
print("V intercept:", round(v_intercept, 6))
print("V test MAE:", round(v_mae, 3), "MW")

# 10. Model comparison
print("\n--- MODEL COMPARISON ---")
print("Baseline:", round(baseline_mae, 3), "MW")
print("AT model:", round(at_mae, 3), "MW")
print("V model:", round(v_mae, 3), "MW")

# 11. Hand-check first test observation
test_position = 0
file_line = 7656

v_value = X_test.iloc[test_position]["V"]
true_pe = y_test.iloc[test_position]

predicted_pe = v_slope * v_value + v_intercept
absolute_error = abs(true_pe - predicted_pe)

print("\n--- HAND CHECK ---")
print("File row:", file_line)
print("V value:", v_value)
print("Predicted PE:", round(predicted_pe, 2), "MW")
print("True PE:", true_pe, "MW")
print("Absolute error:", round(absolute_error, 2), "MW")

# 12. Software versions
print("\n--- SOFTWARE VERSIONS ---")
print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
