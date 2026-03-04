import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, roc_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib to show plots inline
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def plot_target_distribution(y_counts, title="Target Variable Distribution"):
    """Plot target variable distribution"""
    plt.figure(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e']
    bars = plt.bar(y_counts.index, y_counts.values, color=colors)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Readmission Status', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('target_distribution.png', dpi=100, bbox_inches='tight')
    plt.show()

def plot_feature_importance(feature_importance_df, top_n=20, title="Top Feature Importances"):
    """Plot feature importance"""
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
    
    bars = plt.barh(range(len(top_features)), top_features['Importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (importance, feature) in enumerate(zip(top_features['Importance'], top_features['Feature'])):
        plt.text(importance + 0.001, i, f'{importance:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix_custom(cm, class_names, title='Confusion Matrix'):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.show()

def plot_roc_curve(y_test, y_pred_proba, title='ROC Curve'):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_curve.png', dpi=100, bbox_inches='tight')
    plt.show()

def plot_model_comparison(results_df, metric='AUC_ROC', title='Model Comparison'):
    """Plot model comparison"""
    plt.figure(figsize=(10, 6))
    
    models = results_df['Model']
    values = results_df[metric]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = plt.bar(models, values, color=colors[:len(models)])
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.ylim([0, max(values) * 1.1])
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=11)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
    plt.show()

def clean_dataset(df, target_col, save_cleaned=True):
    """
    Comprehensive data cleaning function
    Removes missing values, duplicates, and performs data validation
    Returns cleaned dataframe and saves it to file
    """
    print("\n" + "="*60)
    print("DATA CLEANING AND PREPROCESSING")
    print("="*60)
    
    original_shape = df.shape
    print(f"Original dataset shape: {original_shape}")
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    # 1. REMOVE DUPLICATES
    print("\n1. CHECKING FOR DUPLICATES...")
    duplicate_rows = df_clean.duplicated().sum()
    if duplicate_rows > 0:
        print(f"   Found {duplicate_rows} duplicate rows")
        df_clean = df_clean.drop_duplicates()
        print(f"   Removed duplicates. New shape: {df_clean.shape}")
    else:
        print("   No duplicate rows found")
    
    # 2. CONVERT COLUMN DATA TYPES BEFORE HANDLING MISSING VALUES
    print("\n2. CONVERTING DATA TYPES...")
    
    # Identify which columns should be numeric based on name or content
    potential_numeric_cols = ['Age', 'NumberOfDiagnoses', 'NumberOfProcedures', 
                             'NumberOfMedications', 'PriorAdmissions', 'TimeInHospital',
                             'NumberOfLabProcedures', 'NumberOfEmergencyVisits', 
                             'ComorbidityCount', 'BMI']
    
    for col in potential_numeric_cols:
        if col in df_clean.columns:
            try:
                # Convert to numeric, forcing errors to NaN
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                print(f"   ✓ Converted '{col}' to numeric")
            except:
                print(f"   ⚠️  Could not convert '{col}' to numeric")
    
    # 3. HANDLE MISSING VALUES
    print("\n3. CHECKING FOR MISSING VALUES...")
    missing_values = df_clean.isnull().sum()
    missing_cols = missing_values[missing_values > 0]
    
    if len(missing_cols) > 0:
        print("   Columns with missing values:")
        for col, count in missing_cols.items():
            percent = (count / len(df_clean)) * 100
            print(f"     {col}: {count} missing ({percent:.1f}%)")
        
        # Strategy for handling missing values
        print("\n   HANDLING MISSING VALUES:")
        
        for col in missing_cols.index:
            if col == target_col:
                print(f"     ✗ Removing rows with missing target values in '{col}'")
                df_clean = df_clean.dropna(subset=[col])
            elif df_clean[col].dtype == 'object':  # Categorical column
                mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(mode_value)
                print(f"     ✓ Filled missing categorical '{col}' with mode: '{mode_value}'")
            elif pd.api.types.is_numeric_dtype(df_clean[col]):  # Numerical column
                median_value = df_clean[col].median(skipna=True)
                df_clean[col] = df_clean[col].fillna(median_value)
                print(f"     ✓ Filled missing numerical '{col}' with median: {median_value:.2f}")
            else:
                # For columns that couldn't be converted to numeric
                mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(mode_value)
                print(f"     ✓ Filled missing mixed-type '{col}' with mode: '{mode_value}'")
    else:
        print("   No missing values found")
    
    # 4. CHECK FOR INVALID VALUES IN KEY COLUMNS
    print("\n4. VALIDATING DATA TYPES AND VALUES...")
    
    # Ensure target column is clean
    if target_col in df_clean.columns:
        # Remove any whitespace from target values
        df_clean[target_col] = df_clean[target_col].astype(str).str.strip()
        unique_targets = df_clean[target_col].unique()
        print(f"   Target column '{target_col}' unique values: {unique_targets}")
    
    # Check for invalid numerical values (negative counts, etc.)
    count_columns = ['NumberOfDiagnoses', 'NumberOfProcedures', 'NumberOfMedications', 
                     'PriorAdmissions', 'TimeInHospital', 'NumberOfLabProcedures', 
                     'NumberOfEmergencyVisits', 'ComorbidityCount']
    
    for col in count_columns:
        if col in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean[col]):
            # Replace negative values with 0
            negative_count = (df_clean[col] < 0).sum()
            if negative_count > 0:
                print(f"   ✓ Fixed {negative_count} negative values in '{col}'")
                df_clean.loc[df_clean[col] < 0, col] = 0
    
    # 5. CHECK FOR UNUSUAL VALUES IN AGE
    if 'Age' in df_clean.columns and pd.api.types.is_numeric_dtype(df_clean['Age']):
        # Remove unrealistic ages
        realistic_age_mask = (df_clean['Age'] >= 0) & (df_clean['Age'] <= 120)
        unrealistic_count = (~realistic_age_mask).sum()
        if unrealistic_count > 0:
            print(f"   ✓ Removed {unrealistic_count} unrealistic age values")
            df_clean = df_clean[realistic_age_mask]
    
    # 6. FINAL CLEANUP - Remove any remaining rows with missing values
    final_missing = df_clean.isnull().sum().sum()
    if final_missing > 0:
        missing_rows = df_clean.isnull().any(axis=1).sum()
        print(f"\n   Removing {missing_rows} rows with remaining missing values")
        df_clean = df_clean.dropna()
    
    # 7. SAVE CLEANED DATASET
    if save_cleaned:
        cleaned_filepath = "cleaned_patient_readmission_data.csv"
        df_clean.to_csv(cleaned_filepath, index=False)
        print(f"\n   ✓ Cleaned dataset saved to: {cleaned_filepath}")
        if os.path.exists(cleaned_filepath):
            print(f"   ✓ File size: {os.path.getsize(cleaned_filepath) / 1024:.1f} KB")
    
    # 8. SUMMARY
    print("\n5. CLEANING SUMMARY:")
    print(f"   Original dataset: {original_shape[0]} rows, {original_shape[1]} columns")
    print(f"   Cleaned dataset: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    print(f"   Rows removed: {original_shape[0] - df_clean.shape[0]}")
    print(f"   Data retention: {df_clean.shape[0]/original_shape[0]*100:.1f}%")
    
    return df_clean

def analyze_dataset(df, target_col):
    """Analyze the dataset for class distribution and feature characteristics"""
    print("\n" + "="*60)
    print("DATASET ANALYSIS")
    print("="*60)
    
    # Basic info
    print(f"Total patients: {len(df)}")
    print(f"Total features: {len(df.columns) - 1}")  # Excluding target
    
    # Target distribution
    print("\n1. TARGET VARIABLE DISTRIBUTION:")
    target_counts = df[target_col].value_counts()
    target_percent = df[target_col].value_counts(normalize=True) * 100
    
    for label in target_counts.index:
        print(f"   {label}: {target_counts[label]} patients ({target_percent[label]:.1f}%)")
    
    # Class imbalance ratio
    if len(target_counts) == 2:
        imbalance_ratio = target_counts.max() / target_counts.min()
        print(f"\n   Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
        if imbalance_ratio > 2:
            print(f"   ⚠️  Significant class imbalance detected")
    
    # Feature types
    print("\n2. FEATURE TYPES:")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"   Categorical features: {len(categorical_cols)}")
    print(f"   Numerical features: {len(numerical_cols)}")
    
    # Check for missing values in cleaned dataset
    print("\n3. DATA QUALITY CHECK (CLEANED DATASET):")
    missing_after = df.isnull().sum().sum()
    duplicates_after = df.duplicated().sum()
    
    if missing_after == 0:
        print("   ✓ No missing values")
    else:
        print(f"   ⚠️  {missing_after} missing values still present")
    
    if duplicates_after == 0:
        print("   ✓ No duplicate rows")
    else:
        print(f"   ⚠️  {duplicates_after} duplicate rows still present")
    
    # Data types summary
    print("\n4. DATA TYPE DISTRIBUTION:")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"   {dtype}: {count} columns")
    
    return categorical_cols, numerical_cols, target_counts

def create_features(df):
    """Create additional features through feature engineering"""
    print("\n6. CREATING NEW FEATURES...")
    
    # Copy the dataframe
    df_engineered = df.copy()
    
    # Convert numeric columns if needed
    numeric_cols_to_check = ['Age', 'ComorbidityCount', 'TimeInHospital', 
                            'NumberOfDiagnoses', 'PriorAdmissions', 
                            'NumberOfEmergencyVisits']
    
    for col in numeric_cols_to_check:
        if col in df_engineered.columns:
            df_engineered[col] = pd.to_numeric(df_engineered[col], errors='coerce')
    
    # Interaction features (only if columns exist and are numeric)
    if 'Age' in df_engineered.columns and 'ComorbidityCount' in df_engineered.columns:
        if pd.api.types.is_numeric_dtype(df_engineered['Age']) and pd.api.types.is_numeric_dtype(df_engineered['ComorbidityCount']):
            df_engineered['Age_Comorbidity'] = df_engineered['Age'] * df_engineered['ComorbidityCount']
            print("   Created: Age_Comorbidity")
    
    if 'TimeInHospital' in df_engineered.columns and 'NumberOfDiagnoses' in df_engineered.columns:
        if pd.api.types.is_numeric_dtype(df_engineered['TimeInHospital']) and pd.api.types.is_numeric_dtype(df_engineered['NumberOfDiagnoses']):
            df_engineered['LOS_Diagnoses'] = df_engineered['TimeInHospital'] * df_engineered['NumberOfDiagnoses']
            print("   Created: LOS_Diagnoses")
    
    if 'PriorAdmissions' in df_engineered.columns and 'NumberOfEmergencyVisits' in df_engineered.columns:
        if pd.api.types.is_numeric_dtype(df_engineered['PriorAdmissions']) and pd.api.types.is_numeric_dtype(df_engineered['NumberOfEmergencyVisits']):
            df_engineered['TotalPreviousVisits'] = df_engineered['PriorAdmissions'] + df_engineered['NumberOfEmergencyVisits']
            print("   Created: TotalPreviousVisits")
    
    # Risk score (simple) - convert to numeric first
    risk_factors = []
    risk_cols = ['HasDiabetes', 'HasHeartDisease', 'HasHypertension', 'HasRenalDisease']
    
    for col in risk_cols:
        if col in df_engineered.columns:
            # Convert to numeric if it's a string representation of numbers
            df_engineered[col] = pd.to_numeric(df_engineered[col], errors='coerce').fillna(0)
            risk_factors.append(df_engineered[col])
    
    if risk_factors:
        df_engineered['RiskFactorCount'] = sum(risk_factors)
        print("   Created: RiskFactorCount")
    
    # Age groups (only if Age is numeric)
    if 'Age' in df_engineered.columns and pd.api.types.is_numeric_dtype(df_engineered['Age']):
        bins = [0, 40, 60, 80, 120]
        labels = ['Young', 'Middle', 'Senior', 'Elderly']
        df_engineered['AgeGroup'] = pd.cut(df_engineered['Age'], bins=bins, labels=labels, right=False)
        print("   Created: AgeGroup")
    
    # Handle any NaN values created during feature engineering
    engineered_cols = ['Age_Comorbidity', 'LOS_Diagnoses', 'TotalPreviousVisits', 'RiskFactorCount']
    for col in engineered_cols:
        if col in df_engineered.columns:
            df_engineered[col] = df_engineered[col].fillna(0)
    
    print(f"   Total features after engineering: {len(df_engineered.columns)}")
    return df_engineered

# Main execution
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PATIENT READMISSION PREDICTION WITH DATA CLEANING")
    print("="*60)
    
    # Configuration
    DATA_PATH = r"C:\Users\akono\OneDrive\Desktop\AI_health\patient_readmission_data.csv"
    CLEANED_DATA_PATH = "cleaned_patient_readmission_data.csv"
    FORCE_CLEAN = False  # Set to False to use existing cleaned file
    
    # Load original dataset first to identify target column
    print(f"\nDataset source: {DATA_PATH}")
    print(f"Cleaned dataset will be saved as: {CLEANED_DATA_PATH}")
    
    try:
        # Load a sample to identify columns
        df_sample = pd.read_csv(DATA_PATH, nrows=5)
        print(f"\nSample of dataset (first 5 rows):")
        print(df_sample.head())
        
        # Auto-detect target column
        possible_targets = ["Readmitted", "Readmitted30Days", "readmitted", "target"]
        target_col = next((col for col in possible_targets if col in df_sample.columns), None)
        
        if target_col is None:
            print(f"\n✗ No common target column name found.")
            print("Available columns:", list(df_sample.columns))
            print("\nPlease specify the target column name from the list above:")
            target_col = input("Target column name: ").strip()
            if target_col not in df_sample.columns:
                raise ValueError(f"Column '{target_col}' not found in dataset.")
        
        print(f"\n✓ Target column identified: {target_col}")
        
    except Exception as e:
        print(f"Error reading dataset: {e}")
        print("Using default target column: Readmitted30Days")
        target_col = "Readmitted30Days"
    
    # Check if we should use existing cleaned data
    if not FORCE_CLEAN and os.path.exists(CLEANED_DATA_PATH):
        print(f"\n✓ Found existing cleaned dataset: {CLEANED_DATA_PATH}")
        print("  Loading cleaned dataset...")
        df_cleaned = pd.read_csv(CLEANED_DATA_PATH)
        print(f"  Loaded {len(df_cleaned)} rows, {len(df_cleaned.columns)} columns")
        
        # Verify target column exists in cleaned dataset
        if target_col not in df_cleaned.columns:
            print(f"✗ Warning: Target column '{target_col}' not found in cleaned dataset")
            print("  Loading original dataset for cleaning...")
            df_original = pd.read_csv(DATA_PATH)
            df_cleaned = clean_dataset(df_original, target_col, save_cleaned=True)
    else:
        print(f"\n✓ Loading original dataset from: {DATA_PATH}")
        df_original = pd.read_csv(DATA_PATH)
        print(f"  Original dataset shape: {df_original.shape}")
        df_cleaned = clean_dataset(df_original, target_col, save_cleaned=True)
    
    # Save a backup of the cleaned dataset
    df_cleaned_backup = df_cleaned.copy()
    print(f"\n✓ Cleaned dataset ready for processing: {df_cleaned.shape}")
    
    # Analyze cleaned dataset
    categorical_cols, numerical_cols, target_counts = analyze_dataset(df_cleaned, target_col)
    
    # Plot target distribution
    print("\n📊 Plotting target distribution...")
    plot_target_distribution(target_counts)
    
    # Feature engineering
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    df_engineered = create_features(df_cleaned)
    
    # Save engineered dataset
    ENGINEERED_DATA_PATH = "engineered_patient_readmission_data.csv"
    df_engineered.to_csv(ENGINEERED_DATA_PATH, index=False)
    print(f"\n✓ Engineered dataset saved as: {ENGINEERED_DATA_PATH}")
    
    # Update categorical and numerical columns after feature engineering
    categorical_cols = df_engineered.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    numerical_cols = df_engineered.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\n✓ Using engineered dataset with {len(df_engineered)} patients and {len(df_engineered.columns)} features")
    
    # Check for any remaining missing values
    remaining_missing = df_engineered.isnull().sum().sum()
    if remaining_missing > 0:
        print(f"\n⚠️  Warning: {remaining_missing} missing values still present after engineering")
        print("  Filling with appropriate values...")
        
        for col in df_engineered.columns:
            if df_engineered[col].isnull().sum() > 0:
                if df_engineered[col].dtype == 'object':
                    df_engineered[col] = df_engineered[col].fillna('Unknown')
                elif pd.api.types.is_numeric_dtype(df_engineered[col]):
                    df_engineered[col] = df_engineered[col].fillna(df_engineered[col].median())
    
    # Separate features and target
    X = df_engineered.drop(columns=[target_col])
    y = df_engineered[target_col]
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))
    
    print(f"\nTarget encoding: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # One-hot encode categorical columns - REDUCE FEATURES
    print(f"\n7. PREPROCESSING DATA...")
    print(f"   One-hot encoding {len(categorical_cols)} categorical columns...")
    
    # Handle categorical columns - convert to dummy variables
    # Limit to top categories to avoid too many features
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # If too many features, reduce dimensionality
    if X_encoded.shape[1] > 500:
        print(f"   ⚠️  Too many features ({X_encoded.shape[1]}). Reducing...")
        # Keep only columns with variance
        from sklearn.feature_selection import VarianceThreshold
        selector = VarianceThreshold(threshold=0.01)
        X_encoded = pd.DataFrame(selector.fit_transform(X_encoded), 
                                 columns=X_encoded.columns[selector.get_support()])
        print(f"   Reduced to {X_encoded.shape[1]} features")
    
    print(f"   Features after encoding: {X_encoded.shape[1]}")
    
    # Save the encoded features for future use
    ENCODED_DATA_PATH = "encoded_patient_readmission_data.csv"
    X_encoded_with_target = X_encoded.copy()
    X_encoded_with_target[target_col] = y_encoded
    X_encoded_with_target.to_csv(ENCODED_DATA_PATH, index=False)
    print(f"   ✓ Encoded dataset saved as: {ENCODED_DATA_PATH}")
    
    # Split dataset with stratification
    print("\n8. SPLITTING DATA...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"   Training set: {X_train.shape[0]} patients")
    print(f"   Test set: {X_test.shape[0]} patients")
    print(f"   Features: {X_train.shape[1]}")
    
    # Scale numerical features
    scaler = StandardScaler()
    num_cols_encoded = [col for col in X_train.columns if col in numerical_cols]
    if num_cols_encoded:
        X_train[num_cols_encoded] = scaler.fit_transform(X_train[num_cols_encoded])
        X_test[num_cols_encoded] = scaler.transform(X_test[num_cols_encoded])
        print(f"   Scaled {len(num_cols_encoded)} numerical features")
    
    print("\n" + "="*60)
    print("MODEL TRAINING WITH CLASS IMBALANCE HANDLING")
    print("="*60)
    
    # Define models with different strategies - SIMPLER GRID FOR SPEED
    models = {
        'RandomForest_Balanced': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
        'RandomForest_Weighted': RandomForestClassifier(random_state=42, class_weight={0: 1, 1: 3}, n_jobs=-1),
    }
    
    # Simplified hyperparameter grids for faster training
    param_grids = {
        'RandomForest_Balanced': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
        },
        'RandomForest_Weighted': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
        }
    }
    
    # Skip GradientBoosting if too many features
    if X_train.shape[1] < 500:
        models['GradientBoosting'] = GradientBoostingClassifier(random_state=42)
        param_grids['GradientBoosting'] = {
            'n_estimators': [100],
            'max_depth': [3, 5],
            'learning_rate': [0.1, 0.2],
        }
    
    best_models = {}
    results = []
    
    for model_name, model in models.items():
        print(f"\n9. TRAINING {model_name}...")
        
        # Grid search with timeout protection
        try:
            grid_search = GridSearchCV(
                model,
                param_grids[model_name],
                cv=3,
                n_jobs=1,  # Reduce to 1 to prevent memory issues
                verbose=1,
                scoring='roc_auc',
                error_score='raise'
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Predictions
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc_roc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            # Store results
            results.append({
                'Model': model_name,
                'Best_Params': str(grid_search.best_params_),
                'CV_Score': grid_search.best_score_,
                'Accuracy': accuracy,
                'F1_Score': f1,
                'AUC_ROC': auc_roc
            })
            
            best_models[model_name] = best_model
            
            print(f"   Best parameters: {grid_search.best_params_}")
            print(f"   Best CV Score (AUC-ROC): {grid_search.best_score_:.4f}")
            print(f"   Test Accuracy: {accuracy:.4f}")
            print(f"   Test F1-Score: {f1:.4f}")
            if auc_roc:
                print(f"   Test AUC-ROC: {auc_roc:.4f}")
            
        except Exception as e:
            print(f"   Error training {model_name}: {str(e)}")
            continue
    
    # Compare all models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    if results:
        results_df = pd.DataFrame(results)
        print("\nPerformance Summary:")
        print(results_df.to_string(index=False))
        
        # Plot model comparison
        print("\n📊 Plotting model comparison...")
        plot_model_comparison(results_df, metric='AUC_ROC')
        
        # Find best model based on AUC-ROC or F1
        if 'AUC_ROC' in results_df.columns and results_df['AUC_ROC'].notna().any():
            best_model_name = results_df.loc[results_df['AUC_ROC'].idxmax(), 'Model']
        else:
            best_model_name = results_df.loc[results_df['F1_Score'].idxmax(), 'Model']
        
        print(f"\n✓ Best performing model: {best_model_name}")
        
        # Get the best model for detailed analysis
        if best_model_name in best_models:
            best_model = best_models[best_model_name]
            
            # Make predictions on test set with best model
            y_pred_best = best_model.predict(X_test)
            y_pred_proba_best = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
            
            cm = confusion_matrix(y_test, y_pred_best)
            
            print("\n" + "="*60)
            print("FINAL MODEL PERFORMANCE")
            print("="*60)
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))
            
            # Plot confusion matrix
            print("\n📊 Plotting confusion matrix...")
            plot_confusion_matrix_custom(cm, label_encoder.classes_, title=f'Confusion Matrix - {best_model_name}')
            
            print("\nConfusion Matrix:")
            print(cm)
            
            # Calculate clinical metrics
            sensitivity = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
            specificity = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
            precision = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0
            
            print(f"\nCLINICAL METRICS:")
            print(f"  Sensitivity (Readmission Detection): {sensitivity:.1%}")
            print(f"  Specificity (Non-readmission Accuracy): {specificity:.1%}")
            print(f"  Precision (Correct Readmission Predictions): {precision:.1%}")
            print(f"  Overall Accuracy: {accuracy_score(y_test, y_pred_best):.1%}")
            
            # Plot ROC curve if we have probability predictions
            if y_pred_proba_best is not None:
                print("\n📊 Plotting ROC curve...")
                plot_roc_curve(y_test, y_pred_proba_best, title=f'ROC Curve - {best_model_name}')
            
            # Feature importance
            if hasattr(best_model, 'feature_importances_'):
                importances = best_model.feature_importances_
                
                # Create feature importance DataFrame
                feature_importance_df = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                print("\n" + "="*60)
                print("TOP 20 MOST IMPORTANT FEATURES")
                print("="*60)
                print(feature_importance_df.head(20).to_string(index=False))
                
                # Plot feature importance
                print("\n📊 Plotting feature importance...")
                plot_feature_importance(feature_importance_df, top_n=20, title=f'Top 20 Feature Importances - {best_model_name}')
                
                # Save feature importance to file
                feature_importance_df.to_csv('feature_importance.csv', index=False)
                print("\n✓ Feature importance saved as 'feature_importance.csv'")
            
            # Save the best model
            model_data = {
                'model': best_model,
                'label_encoder': label_encoder,
                'feature_names': X_train.columns.tolist(),
                'scaler': scaler,
                'categorical_cols': categorical_cols,
                'numerical_cols': numerical_cols,
                'model_name': best_model_name,
                'performance': results_df[results_df['Model'] == best_model_name].iloc[0].to_dict(),
                'cleaned_data_path': CLEANED_DATA_PATH,
                'engineered_data_path': ENGINEERED_DATA_PATH,
                'encoded_data_path': ENCODED_DATA_PATH
            }
            
            joblib.dump(model_data, "best_readmission_model.pkl")
            print("\n✓ Best model saved as 'best_readmission_model.pkl'")
            
            # Save performance metrics
            results_df.to_csv('model_performance_comparison.csv', index=False)
            print("✓ Model comparison saved as 'model_performance_comparison.csv'")
            
            # Generate comprehensive report
            with open('final_model_report.txt', 'w') as f:
                f.write("="*80 + "\n")
                f.write("FINAL MODEL PERFORMANCE REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write("DATASET INFORMATION:\n")
                f.write("-"*40 + "\n")
                f.write(f"Original data: {DATA_PATH}\n")
                f.write(f"Cleaned data: {CLEANED_DATA_PATH} ({df_cleaned_backup.shape[0]} rows)\n")
                f.write(f"Engineered data: {ENGINEERED_DATA_PATH} ({df_engineered.shape[0]} rows)\n")
                f.write(f"Encoded data: {ENCODED_DATA_PATH} ({X_encoded.shape[1]} features)\n")
                f.write(f"Training set: {X_train.shape[0]} patients\n")
                f.write(f"Test set: {X_test.shape[0]} patients\n\n")
                
                f.write(f"BEST MODEL: {best_model_name}\n")
                f.write(f"Best Parameters: {best_model.get_params()}\n\n")
                
                f.write("PERFORMANCE METRICS:\n")
                f.write("-"*40 + "\n")
                f.write(f"Accuracy: {accuracy_score(y_test, y_pred_best):.1%}\n")
                f.write(f"Sensitivity (Recall for Readmissions): {sensitivity:.1%}\n")
                f.write(f"Specificity: {specificity:.1%}\n")
                f.write(f"Precision: {precision:.1%}\n")
                if y_pred_proba_best is not None:
                    f.write(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba_best):.1%}\n\n")
                
                f.write("CONFUSION MATRIX:\n")
                f.write("-"*40 + "\n")
                f.write(str(cm) + "\n\n")
                
                f.write("CLASSIFICATION REPORT:\n")
                f.write("-"*40 + "\n")
                f.write(classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))
            
            print("✓ Final report saved as 'final_model_report.txt'")
            
            # Recommendations based on performance
            print("\n" + "="*60)
            print("RECOMMENDATIONS")
            print("="*60)
            
            if sensitivity < 0.5:
                print("⚡ HIGH RISK: Model misses more than 50% of readmissions")
                print("   Actions needed:")
                print("   1. Collect more data on readmitted patients")
                print("   2. Add more predictive features")
                print("   3. Try advanced models (XGBoost, Neural Networks)")
                print("   4. Consider ensemble methods")
            elif sensitivity < 0.7:
                print("⚠️  MODERATE PERFORMANCE: Model detects 50-70% of readmissions")
                print("   Suggestions:")
                print("   1. Fine-tune hyperparameters further")
                print("   2. Add interaction features")
                print("   3. Use feature selection to remove noise")
            else:
                print("✅ GOOD PERFORMANCE: Model detects >70% of readmissions")
                print("   Next steps:")
                print("   1. Validate on external dataset")
                print("   2. Deploy for pilot testing")
                print("   3. Monitor performance over time")
    
    else:
        print("✗ No models were successfully trained.")
        print("\nTroubleshooting tips:")
        print("1. Check if scikit-learn is properly installed")
        print("2. Verify your cleaned dataset has sufficient data")
        print("3. Ensure target column has at least 2 classes")
    
    print("\n" + "="*60)
    print("PROCESS COMPLETE - VISUALIZATIONS CREATED")
    print("="*60)
    print("\n📁 DATASET FILES CREATED:")
    print(f"1. {CLEANED_DATA_PATH} - Cleaned dataset (no duplicates/missing values)")
    print(f"2. {ENGINEERED_DATA_PATH} - Dataset with engineered features")
    print(f"3. {ENCODED_DATA_PATH} - Encoded dataset ready for modeling")
    
    print("\n📊 MODEL FILES CREATED:")
    print("1. best_readmission_model.pkl - Trained model")
    print("2. feature_importance.csv - Feature rankings")
    print("3. model_performance_comparison.csv - Model comparison")
    print("4. final_model_report.txt - Comprehensive report")
    
    print("\n📈 VISUALIZATION FILES CREATED:")
    print("1. target_distribution.png - Target variable distribution")
    print("2. model_comparison.png - Model performance comparison")
    print("3. confusion_matrix.png - Confusion matrix")
    print("4. roc_curve.png - ROC curve (if available)")
    print("5. feature_importance.png - Feature importance plot")
    
    print("\n✅ NEXT STEPS:")
    print(f"1. Use '{CLEANED_DATA_PATH}' for all future analysis")
    print("2. Load 'best_readmission_model.pkl' for predictions")
    print("3. Review the PNG images for visual insights")