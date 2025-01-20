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

import pickle
import joblib
from pathlib import Path

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

def prepare_model_data(ad_df, INDUSTRY_SPECIFIC_AWARENESS, BRAND_SPECIFIC_AWARENESS):
    columns = ['superlative_count', 'comparative_count', 'uniqueness_count', 'total_bdm_terms_count', 'total_bdm_terms_pct', 'num_adj_noun_pairs']

    if INDUSTRY_SPECIFIC_AWARENESS:
        columns.append('product_cat_keyword_similarity')

    if BRAND_SPECIFIC_AWARENESS:
        columns.append('product_brand_keyword_similarity')

    features = ad_df[columns]
    # Extract target variable
    # target = ad_df['BDM']
    
    return features
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
    - predictions: DataFrame with prediction categories (TP, TN, FP, FN)
    """
    results = []
    predictions = pd.DataFrame()
    
    for name, model in trained_models.items():
        print(f"Evaluating {name}...")
        
        # Get predictions
        y_pred = model.predict(X)
        
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
        
        predictions[f'{name}_result'] = pred_categories
        
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
def plot_confusion_matrices(X, y, models):
    """
    Plot confusion matrices with percentages for each model.
    
    Parameters:
    - X: Feature matrix
    - y: Target variable
    - models: Dictionary of tuned models
    """
    n_models = len(models)
    fig, axes = plt.subplots(
        nrows=(n_models + 1) // 2, 
        ncols=2, 
        figsize=(16, 4 * ((n_models + 1) // 2))
    )
    axes = axes.ravel()  # Flatten axes array
    
    for i, (name, model) in enumerate(models.items()):
        # Predict using the model
        y_pred = model.predict(X)
        
        # Compute confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Convert to percentages
        total_negatives = cm[0, 0] + cm[0, 1]  # TN + FN
        total_positives = cm[1, 0] + cm[1, 1]  # TP + FP
        
        percentage_cm = np.array([
            [cm[0, 0] / total_negatives * 100 if total_negatives > 0 else 0,  # TN %
             cm[0, 1] / total_negatives * 100 if total_negatives > 0 else 0],  # FN %
            [cm[1, 0] / total_positives * 100 if total_positives > 0 else 0,  # FP %
             cm[1, 1] / total_positives * 100 if total_positives > 0 else 0]   # TP %
        ])
        
        # Plot confusion matrix
        sns.heatmap(
            percentage_cm, 
            annot=True, 
            fmt='.1f', 
            cmap='Blues', 
            ax=axes[i], 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive']
        )
        axes[i].set_title(f'{name} Confusion Matrix (Percentages)')
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

def display_model_results(data, target, models, results_df):

    print("Cross-Validation Results:\n")
    display(Markdown(results_df.to_markdown(index=False)))
    plot_confusion_matrices(data, target, models)
    plot_correlation_with_target(data, target)