import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
# Prepare features
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from sklearn.metrics import confusion_matrix, roc_curve, auc
# import gridsearchcv
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    roc_auc_score, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    confusion_matrix,
    classification_report
)

import shap

def remove_unwanted_columns(df):
  # Store original columns before removal
  original_columns = df.columns.tolist()

  # Remove columns which aren't numbers or categorical
  df = df.select_dtypes(include=['number', 'category'])
  columns_to_remove = [
        'transcript_superlative_count',
        'transcript_comparative_count',
        'transcript_uniqueness_count',
        'transcript_total_bdm_terms_count',
        'ocr_text_superlative_count',
        'ocr_text_comparative_count',
        'ocr_text_uniqueness_count',
        'ocr_text_total_bdm_terms_count',
        'transcript_num_comparisons',
        'ocr_text_num_comparisons',
        'product_cat_name',
        'brand',
        'product_brand_name'
    ]
  # remove these if they are in the dataframe
  df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

  # Determine which columns were removed
  removed_columns = [col for col in original_columns if col not in df.columns]

  # Display removed columns
  print("Removed columns:", removed_columns)
    # print categorical columns
  print(f"Categorical columns: {df.select_dtypes(include=['category']).columns}")
  print(f"Integer columns: {df.select_dtypes(include=['int']).columns}")
  print(f"Float columns: {df.select_dtypes(include=['float']).columns}")
  # print columns with all other types
  print(f"Other columns: {df.select_dtypes(include=['object']).columns}")
  return df


def get_base_models():
    """Return dictionary of base model configurations"""
    return {
        'Majority Classifier': DummyClassifier(),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Support Vector Machine': SVC(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'AdaBoost': AdaBoostClassifier(random_state=42),
    }


def get_param_distributions():
    """Return parameter distributions for RandomizedSearchCV"""
    return {
        'Logistic Regression': {
            'C': uniform(0.1, 10.0),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        },
        'Decision Tree': {
            'max_depth': randint(3, 10),
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10)
        },
        'Random Forest': {
            'n_estimators': randint(50, 300),
            'max_depth': randint(3, 10),
            'min_samples_split': randint(2, 20)
        },
        'Support Vector Machine': {
            'C': uniform(0.1, 20.0),
            'kernel': ['rbf', 'linear']
        },
        'Gradient Boosting': {
            'n_estimators': randint(50, 300),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(3, 10)
        },
        'Dummy Classifier': {
            'strategy': ['stratified']
        }
    }

def tune_models(X, y, models, param_distributions, cv=5, n_iter=20):
    """Perform RandomizedSearchCV on specified models"""
    tuned_models = {}
    
    for name, model in models.items():
        if name in param_distributions:
            print(f"\nTuning {name}...")
            random_search = RandomizedSearchCV(
                model,
                param_distributions[name],
                n_iter=n_iter,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=42
            )
            random_search.fit(X, y)
            tuned_models[name] = random_search.best_estimator_
            print(f"Best parameters: {random_search.best_params_}")
            print(f"Best score: {random_search.best_score_:.3f}")
        else:
            tuned_models[name] = model
            
    return tuned_models


from pathlib import Path
import joblib
def save_models(INDUSTRY_SPECIFIC_AWARENESS, BRAND_SPECIFIC_AWARENESS, trained_models, output_dir='trained_models'):
    """
    Save trained models to files
    
    Parameters:
    - INDUSTRY_SPECIFIC_AWARENESS: Boolean flag indicating industry-specific awareness
    - BRAND_SPECIFIC_AWARENESS: Boolean flag indicating brand-specific awareness
    - trained_models: Dictionary of trained model instances
    - output_dir: Directory to save the models (default: 'trained_models')
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for name, model in trained_models.items():
        # Create a safe filename from the model name
        filename = name  # Start with the model's name
        filename += '_industry' if INDUSTRY_SPECIFIC_AWARENESS else ''
        filename += '_brand' if BRAND_SPECIFIC_AWARENESS else ''
        filename += '.pkl'  # Add file extension
        
        # Define the filepath
        filepath = Path(output_dir) / filename
        
        # Print the saving message
        print(f"Saving {name} to {filepath}")
        
        # Save the model
        joblib.dump(model, filepath)
def load_models(INDUSTRY_SPECIFIC_AWARENESS, BRAND_SPECIFIC_AWARENESS, model_dir='trained_models'):
    """
    Load trained models from files, considering awareness flags
    
    Parameters:
    - INDUSTRY_SPECIFIC_AWARENESS: Boolean indicating if models are industry-specific
    - BRAND_SPECIFIC_AWARENESS: Boolean indicating if models are brand-specific
    - model_dir: Directory containing the saved models (default: 'trained_models')
    
    Returns:
    - Dictionary of loaded models
    """
    loaded_models = {}
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        raise FileNotFoundError(f"Directory {model_dir} not found")
    
    # Construct the awareness-based suffix
    suffix = ''
    if INDUSTRY_SPECIFIC_AWARENESS:
        suffix += '_industry'
    if BRAND_SPECIFIC_AWARENESS:
        suffix += '_brand'
    
    # Load models matching the suffix
    print(f"Loading models from {model_dir} with suffix {suffix}")
    for model_file in model_dir.glob(f'*{suffix}.pkl'):
        # Extract the model name by stripping the suffix and converting to title case
        name = model_file.stem
        if suffix:
            name = name[:-len(suffix)]  # Remove the awareness-based suffix
        
        # Check if the model file matches the awareness settings
        if (not INDUSTRY_SPECIFIC_AWARENESS and '_industry' in name) or (not BRAND_SPECIFIC_AWARENESS and '_brand' in name):
            continue  # Skip loading this model
        
        print(f"{name} Loading from {model_file}")
        loaded_models[name] = joblib.load(model_file)
    
    return loaded_models

def train_models(X, y, models, INDUSTRY_SPECIFIC_AWARENESS, BRAND_SPECIFIC_AWARENESS):
    """
    Train all models in the provided dictionary
    
    Parameters:
    - X: Feature matrix
    - y: Target variable
    - models: Dictionary of model instances
    
    Returns:
    - Dictionary of trained models
    """
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X, y)
        trained_models[name] = model
    save_models(INDUSTRY_SPECIFIC_AWARENESS, BRAND_SPECIFIC_AWARENESS, trained_models)
    return trained_models


def evaluate_models(X, y, trained_models, cv=5):
    """
    Evaluate trained models and return results
    
    Parameters:
    - X: Feature matrix
    - y: Target variable
    - trained_models: Dictionary of trained model instances
    - cv: Number of cross-validation folds
    
    Returns:
    - results_df: DataFrame with model performance metrics
    - predictions: Dictionary with actual predictions for each model
    """
    results = []
    predictions = {}  # Changed to store actual predictions
    
    for name, model in trained_models.items():
        print(f"Evaluating {name}...")
        
        # Get predictions
        y_pred = model.predict(X)
        predictions[name] = y_pred  # Store actual predictions
        
        # Create confusion matrix categories
        pred_categories = []
        for true, pred in zip(y, y_pred):
            if true == 0 and pred == 0:
                pred_categories.append('TN')
            elif true == 0 and pred == 1:
                pred_categories.append('FP')
            elif true == 1 and pred == 0:
                pred_categories.append('FN')
            else:  # true == 1 and pred == 1
                pred_categories.append('TP')
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, 
            X, 
            y, 
            cv=cv, 
            scoring=['roc_auc', 'accuracy', 'precision', 'recall']
        )
        
        # Calculate metrics
        results.append({
            'Model': name,
            'ROC AUC (Mean)': np.mean(cv_results['test_roc_auc']),
            'ROC AUC (Std)': np.std(cv_results['test_roc_auc']),
            'Accuracy (Mean)': np.mean(cv_results['test_accuracy']),
            'Accuracy (Std)': np.std(cv_results['test_accuracy']),
            'Precision (Mean)': np.mean(cv_results['test_precision']),
            'Precision (Std)': np.std(cv_results['test_precision']),
            'Recall (Mean)': np.mean(cv_results['test_recall']),
            'Recall (Std)': np.std(cv_results['test_recall'])
        })
    
    results_df = pd.DataFrame(results).sort_values('ROC AUC (Mean)', ascending=False)
    return results_df, predictions


def predict_model(X, trained_models):
    """
    Predict using the trained models
    
    Parameters:
    - X: Feature matrix
    - trained_models: Dictionary of trained model instances
    
    Returns:
    - predictions: DataFrame with predictions from each model
    """
    predictions = pd.DataFrame()
    
    for name, model in trained_models.items():
        print(f"Making predictions with {name}...")
        y_pred = model.predict(X)
        predictions[f'{name}_prediction'] = y_pred
        
    return predictions


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def plot_correlation_with_target(X, y):
    """
    Plot the correlation between features and the target variable.
    
    Parameters:
    - X: Feature matrix (pandas DataFrame)
    - y: Target variable (pandas Series or numpy array)
    """
    # Add target to the feature matrix for correlation calculation
    X_with_target = X.copy()
    X_with_target['target'] = y
    
    # Compute the correlation matrix
    correlation_matrix = X_with_target.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix[['target']], annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation between Features and Target", fontsize=16)
    plt.show()


def predict_for_confusion_matrices(X, models):
    """
    Predict labels for each model to be used in confusion matrices.
    
    Parameters:
    - X: Feature matrix
    - models: Dictionary of tuned models
    
    Returns:
    - predictions: Dictionary of predictions for each model
    """
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(X)
    return predictions

def plot_confusion_matrices(X, y, predictions):
    """
    Plot confusion matrices with percentages and fractions for each model.
    
    Parameters:
    - X: Feature matrix
    - y: Target variable
    - predictions: Dictionary of predictions for each model
    """
    n_models = len(predictions)
    fig, axes = plt.subplots(
        nrows=(n_models + 1) // 2, 
        ncols=2, 
        figsize=(16, 4 * ((n_models + 1) // 2))
    )
    axes = axes.ravel()  # Flatten axes array
    
    for i, (name, y_pred) in enumerate(predictions.items()):
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Convert to percentages and fractions for annotations
        total_negatives = cm[0, 0] + cm[0, 1]  # TN + FN
        total_positives = cm[1, 0] + cm[1, 1]  # TP + FP
        
        annotations = np.array([
            [f"{cm[0, 0] / total_negatives * 100:.1f}%\n({cm[0, 0]}/{total_negatives})" if total_negatives > 0 else "0% (0/0)",  # TN %
             f"{cm[0, 1] / total_negatives * 100:.1f}%\n({cm[0, 1]}/{total_negatives})" if total_negatives > 0 else "0% (0/0)"],  # FN %
            [f"{cm[1, 0] / total_positives * 100:.1f}%\n({cm[1, 0]}/{total_positives})" if total_positives > 0 else "0% (0/0)",  # FP %
             f"{cm[1, 1] / total_positives * 100:.1f}%\n({cm[1, 1]}/{total_positives})" if total_positives > 0 else "0% (0/0)"]   # TP %
        ])
        
        # Plot confusion matrix using numerical data
        sns.heatmap(
            cm, 
            annot=annotations, 
            fmt='', 
            cmap='Blues', 
            ax=axes[i], 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive']
        )
        axes[i].set_title(f'{name}')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    
    # Remove extra subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def analyze_decision_tree(data, target, models):
    print("\nDecision Tree Analysis:")
    simple_decision_tree = models['Decision Tree']
    simple_decision_tree.fit(data, target)

    # Create a new figure for the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(simple_decision_tree, 
              filled=True, 
              feature_names=data.columns, 
              class_names=['No BDM', 'BDM'])
    plt.title("Simple Decision Tree Visualization")
    plt.show()

    # Print feature importance ranking
    importances = simple_decision_tree.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\nFeature Importance Ranking:")
    for f in range(data.shape[1]):
        print(f"{f + 1}. {data.columns[indices[f]]}: {importances[indices[f]]:.3f}")


def display_xai(data, target):
  X = data
  y = target
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  rf = RandomForestClassifier(n_estimators=30, max_depth=20, random_state=0, max_features='sqrt',\
                              class_weight='balanced')
  rf.fit(X_train, y_train)
  y_pred_test = rf.predict(X_test)
  samples = X_train
  explainer = shap.TreeExplainer(rf)
  shap_values = explainer.shap_values(X_train, approximate=False, check_additivity=False)
  shap_values_class_1 = shap_values[:, :, 1]
  shap.summary_plot(shap_values_class_1, X_train)

def display_model_results(data, target, models, results_df, predictions):

    display(Markdown("## Cross-Validation Results:\n"))
    display(results_df)
    display(Markdown("## XAI:\n"))
    display_xai(data, target)
    display(Markdown("## Confusion Matrices:\n"))
    plot_confusion_matrices(data, target, predictions)

def assign_data_types(df):
    boolean_columns = [
    'csr_type',
    'BDM',
    'encoded_emotion',
    'transcript_contains_i',
    'ocr_text_contains_i',
    'transcript_contains_we',
    'ocr_text_contains_we',
    'transcript_contains_you',
    'ocr_text_contains_you',
    'transcript_contains_he',
    'ocr_text_contains_he',
    'transcript_contains_she',
    'ocr_text_contains_she',
    'transcript_contains_it',
    'ocr_text_contains_it',
    'transcript_contains_they',
    ]

    integer_columns = [
        'transcript_superlative_count',
        'transcript_comparative_count',
        'transcript_uniqueness_count',
        'transcript_total_bdm_terms_count',
        'ocr_text_superlative_count',
        'ocr_text_comparative_count',
        'ocr_text_uniqueness_count',
        'ocr_text_total_bdm_terms_count',
        'transcript_num_adj_noun_pairs',
        'ocr_text_num_adj_noun_pairs',
        'transcript_num_comparisons',
        'ocr_text_num_comparisons',
    ]
    float_columns = [
        'transcript_superlative_pct',
        'transcript_comparative_pct',
        'transcript_uniqueness_pct',
        'transcript_total_bdm_terms_pct',
        'ocr_text_superlative_pct',
        'ocr_text_comparative_pct',
        'ocr_text_uniqueness_pct',
        'ocr_text_total_bdm_terms_pct',
        'transcript_product_cat_keywords_similarity',
        'ocr_text_product_cat_keywords_similarity',
        'transcript_product_brand_keywords_similarity',
        'ocr_text_product_brand_keywords_similarity',

    ]
    text_columns = [
        'commercial_number',
    ]

    for col in integer_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)

    for col in float_columns:
        if col in df.columns:
            df[col] = df[col].astype(float)
    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype(int)
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df = df.loc[:, (df.columns.isin(boolean_columns) | df.columns.isin(integer_columns) | df.columns.isin(float_columns) | df.columns.isin(text_columns))]
    return df
