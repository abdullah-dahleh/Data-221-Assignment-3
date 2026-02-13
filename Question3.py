import pandas as pd
from sklearn.model_selection import train_test_split

#Load kidney disease dataset
kidney_disease_data_frame = pd.read_csv("kidney_disease.csv")

#Create feature matrix (all columns except classification)
feature_matrix = kidney_disease_data_frame.drop("classification", axis=1)

#Create label vector
label_vector = kidney_disease_data_frame["classification"]

#Split into 70% training and 30% testing
X_training_data, X_testing_data, y_training_data, y_testing_data = train_test_split(
    feature_matrix,
    label_vector,
    test_size=0.3,
    random_state=42
)

print("Training samples:", X_training_data.shape[0])
print("Testing samples:", X_testing_data.shape[0])

#We separate data to test model on unseen data.
#Training and testing on same data gives misleading results.