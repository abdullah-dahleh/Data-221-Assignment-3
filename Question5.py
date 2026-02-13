import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Load dataset
kidney_disease_data_frame = pd.read_csv("kidney_disease.csv")

#Replace '?' with NaN
kidney_disease_data_frame.replace("?", pd.NA, inplace=True)

#Clean classification column
kidney_disease_data_frame["classification"] = kidney_disease_data_frame["classification"].str.strip()
kidney_disease_data_frame["classification"] = kidney_disease_data_frame["classification"].map({
    "ckd": 1,
    "notckd": 0
})

#Select only numeric medical features
selected_numeric_columns = ["age", "bp", "sod", "pot", "hemo", "pcv", "wc", "rc"]

feature_matrix = kidney_disease_data_frame[selected_numeric_columns]

#Convert to numeric
feature_matrix = feature_matrix.apply(pd.to_numeric, errors="coerce")

#Fill missing values with column means
feature_matrix = feature_matrix.fillna(feature_matrix.mean())

label_vector = kidney_disease_data_frame["classification"]

#Remove rows where label is missing
valid_rows = label_vector.notna()
feature_matrix = feature_matrix[valid_rows]
label_vector = label_vector[valid_rows]

print("Remaining NaN values:", feature_matrix.isna().sum().sum())

#Split data
X_training_data, X_testing_data, y_training_data, y_testing_data = train_test_split(
    feature_matrix,
    label_vector,
    test_size=0.3,
    random_state=42
)

#Test different k values
k_values_list = [1, 3, 5, 7, 9]
accuracy_results_list = []

for k_value in k_values_list:
    knn_model = KNeighborsClassifier(n_neighbors=k_value)
    knn_model.fit(X_training_data, y_training_data)

    predicted_test_labels = knn_model.predict(X_testing_data)
    test_accuracy = accuracy_score(y_testing_data, predicted_test_labels)

    accuracy_results_list.append(test_accuracy)
    print("k =", k_value, "Test Accuracy =", test_accuracy)

#Create results table
accuracy_results_table = pd.DataFrame({
    "k_value": k_values_list,
    "test_accuracy": accuracy_results_list
})

print("\nAccuracy Results Table:")
print(accuracy_results_table)

#Small k can overfit the data.
#Large k can underfit the data.
#Best k has the highest test accuracy.