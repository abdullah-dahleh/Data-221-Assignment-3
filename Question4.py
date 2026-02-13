import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

#Select ONLY numeric columns required by assignment
selected_numeric_columns = [
    "age", "bp", "sod", "pot", "hemo", "pcv", "wc", "rc"
]

feature_matrix = kidney_disease_data_frame[selected_numeric_columns]

#Convert to numeric
feature_matrix = feature_matrix.apply(pd.to_numeric, errors="coerce")

#Fill missing values with column means
feature_matrix = feature_matrix.fillna(feature_matrix.mean())

label_vector = kidney_disease_data_frame["classification"]

#Drop any remaining missing labels
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

#Train KNN
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_training_data, y_training_data)

#Predict
predicted_test_labels = knn_classifier.predict(X_testing_data)

#Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_testing_data, predicted_test_labels))
print("Accuracy:", accuracy_score(y_testing_data, predicted_test_labels))
print("Precision:", precision_score(y_testing_data, predicted_test_labels))
print("Recall:", recall_score(y_testing_data, predicted_test_labels))
print("F1 Score:", f1_score(y_testing_data, predicted_test_labels))

#Only numeric medical features were used.
#Missing values were replaced with column means.
#Recall is important because missing disease cases is serious.