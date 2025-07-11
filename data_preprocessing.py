import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    df.drop_duplicates(inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill missing numeric values with mean
    df.fillna("Unknown", inplace=True)  # Fill missing categorical values with 'Unknown'
    return df

def encode_features(df, categorical_columns):
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def scale_features(df, numerical_columns):
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df, scaler

def split_data(df, target_column, test_size=0.2):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Example usage
if __name__ == "__main__":
    file_path = "obese_data.csv"  # Replace with actual dataset path
    df = load_data(file_path)
    df = clean_data(df)
    categorical_columns = ["food_type", "activity_level"]  # Example categorical columns
    numerical_columns = ["calories", "protein", "fat"]  # Example numerical columns
    df, encoders = encode_features(df, categorical_columns)
    df, scaler = scale_features(df, numerical_columns)
    X_train, X_test, y_train, y_test = split_data(df, target_column="obesity_level")
    
    print("Data preprocessing completed successfully!")

