import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(path: str) -> pd.DataFrame:
    """
    Load the studentInfo.csv file and return as DataFrame.
    """
    df = pd.read_csv(path)
    print(f"✅ Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def preprocess_data(df: pd.DataFrame):
    """
    Clean and preprocess the OULAD studentInfo dataset.
    - Handle missing values
    - Encode categorical features
    - Prepare X (features) and y (target)
    """
    df = df.copy()
    
    # Drop irrelevant columns
    drop_cols = ['id_student', 'code_module', 'code_presentation']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    # Handle missing data
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Map final_result to numeric classes
    result_map = {
        'Withdrawn': 0,
        'Fail': 1,
        'Pass': 2,
        'Distinction': 3
    }
    df['final_result'] = df['final_result'].map(result_map)
    
    # Split features and target
    X = df.drop(columns=['final_result'])
    y = df['final_result']
    
    # One-hot encode categorical features
    cat_cols = X.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_encoded = pd.DataFrame(encoder.fit_transform(X[cat_cols]))
    X_encoded.columns = encoder.get_feature_names_out(cat_cols)
    
    # Combine encoded and numeric features
    X_final = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1)
    
    print(f"✅ Data preprocessed successfully. Final feature shape: {X_final.shape}")
    return X_final, y

def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Train a Random Forest classifier and return the model with metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model trained successfully! Accuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

if __name__ == "__main__":
    df = load_data("OULAD/studentInfo.csv")
    X, y = preprocess_data(df)
    model = train_model(X, y)
