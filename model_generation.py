import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split 
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# def label_encode_columns(dataset, columns):
#     encoded_dataset = dataset.copy()  # make a copy of the dataset to avoid modifying the original
#     for column in columns:
#         encoder = LabelEncoder()
#         encoded_dataset[column] = encoder.fit_transform(encoded_dataset[column])
#     return encoded_dataset

def label_encode_columns(dataset, columns):
    encoded_dataset = dataset.copy()
    encoders = {}
    for column in columns:
        if column not in encoders:
            encoders[column] = LabelEncoder()
        encoder = encoders[column]
        encoded_dataset[column] = encoder.fit_transform(encoded_dataset[column])
    # Save encoder dictionary to a pickle file
    with open('encoder_dict.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    return encoded_dataset


def balance_classes_with_smote(dataset, target_column, majority_minority_ratio=0.75):
    # Separate the feature and target columns
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]

    # Calculate the number of samples we want for the minority class
    minority_class_count = y.value_counts()[1]
    majority_class_count = int(minority_class_count / majority_minority_ratio)

    # Apply SMOTE to oversample the minority class
    smote = SMOTE(sampling_strategy=minority_class_count / majority_class_count)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Combine the resampled feature and target columns into a new DataFrame
    resampled_df = pd.concat([X_resampled, y_resampled], axis=1)
    return resampled_df

def train_xgboost(x_train, y_train, x_val, y_val):
    # Define the hyperparameters to tune
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.5]
    }

    # Define the XGBoost classifier
    xgb = XGBClassifier(random_state=42)

    # Perform cross-validation to tune the hyperparameters
    xgb_cv = GridSearchCV(xgb, param_grid, cv=5)
    xgb_cv.fit(x_train, y_train)

    # Print the best hyperparameters found by cross-validation
    print("Best hyperparameters: ", xgb_cv.best_params_)

    # Use the best hyperparameters to train a new XGBoost classifier
    xgb_tuned = XGBClassifier(**xgb_cv.best_params_, random_state=42)
    xgb_tuned.fit(x_train, y_train)

    # Evaluate the tuned model on the validation data
    accuracy = xgb_tuned.score(x_val, y_val)
    print("Validation accuracy: {:.2f}%".format(accuracy * 100))
    return xgb_tuned

data = pd.read_csv('/Users/nishchalmishra/Desktop/Projects/Datasets/healthcare-dataset-stroke-data.csv')
data['bmi'].fillna(data['bmi'].median(), inplace=True)

df = label_encode_columns(data, ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'])
df2 = balance_classes_with_smote(df, 'stroke')

train_val_df, test = train_test_split(df2, test_size=0.2, random_state=42)
train, val = train_test_split(train_val_df, test_size=0.1, random_state=42)

x_train, y_train = train.drop("stroke", axis=1), train["stroke"]
x_val, y_val = val.drop("stroke", axis=1), val["stroke"]
x_test, y_test = test.drop("stroke", axis=1), test["stroke"]

xgb = train_xgboost(x_train, y_train, x_val, y_val)

# Save the trained model as a pickle file
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(xgb, f)