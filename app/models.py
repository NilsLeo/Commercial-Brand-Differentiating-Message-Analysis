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
from sklearn.preprocessing import LabelBinarizer
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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve


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

def plot_tree_depth_vs_roc_auc_cv(X, y, max_depth=20, cv=5):
    """
    Plots test ROC AUC as a function of tree depth using 5-fold cross-validation.

    Parameters:
    - X: Feature matrix (input data).
    - y: Target vector (binary classification labels).
    - max_depth: Maximum tree depth to evaluate.
    - cv: Number of cross-validation folds (default is 5).
    """
    # Binarize the target labels (in case it's not already binary)
    lb = LabelBinarizer()
    y_bin = lb.fit_transform(y).ravel()  # Flatten to 1D array

    # Store average test ROC AUC scores for each fold at each depth
    test_roc_auc = []

    # Evaluate performance at each tree depth using cross-validation
    for depth in range(1, max_depth + 1):
        # Create the decision tree classifier
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        
        # Perform cross-validation on the test set (the validation fold during cross-validation)
        test_auc_scores = cross_val_score(clf, X, y_bin, cv=cv, scoring='roc_auc')
        
        # Append average ROC AUC score
        test_roc_auc.append(np.mean(test_auc_scores))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_depth + 1), test_roc_auc, label='Test ROC AUC (CV)', marker='o', color='red')
    plt.xlabel('Tree Depth')
    plt.ylabel('ROC AUC')
    plt.title('Decision Tree ROC AUC vs. Depth (5-Fold CV)')
    plt.legend()
    plt.grid(True)
    plt.show()

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
    sns.heatmap(correlation_matrix[['target']], annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, annot_kws={"size": 16})
    plt.title("Correlation between Features and Target", fontsize=16)
    plt.show()




def plot_confusion_matrices(X, y, predictions):
    colors = ['#AFCAE2', '#F1BEA1', '#5694CE', '#E4E1DA', '#A4CD91', '#C6D6C5']
    sns.set_palette(colors)
    """
    Plot confusion matrices with absolute numbers and percentages for each model.
    
    Parameters:
    - X: Feature matrix
    - y: Target variable
    - predictions: Dictionary of predictions for each model
    """
    # Iterate over each model's predictions
    for name, y_pred in predictions.items():
        # Calculate the confusion matrix
        cm = confusion_matrix(y, y_pred)
        # Normalize the confusion matrix to show percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Combine absolute and normalized values for annotation
        labels = (np.asarray(["{0:0.0f}\n({1:.2%})".format(value, value_norm)
                              for value, value_norm in zip(cm.flatten(), cm_normalized.flatten())])
                  .reshape(cm.shape))
        
        # Plotting
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=labels, fmt="", cmap='Blues', annot_kws={"size": 22}) 
        plt.title(f"Confusion Matrix for {name}")
        plt.xlabel('Predicted label')
        plt.ylabel('True label') 
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



def display_xai(data, target, model):
    X = data
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Use the provided model instead of creating a new RandomForestClassifier
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    samples = X_train
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train, approximate=False, check_additivity=False)
    shap_values_class_1 = shap_values[:, :, 1]
    shap.summary_plot(shap_values_class_1, X_train, max_display=12)  # Show only top 15 features


def display_model_results(data, target, models, results_df, predictions):

    display(Markdown("## Cross-Validation Results:\n"))
    display(results_df)
    display(Markdown("## XAI:\n"))
    dct = models['Decision Tree']
    # display the decision tree
    display_decision_tree(data, target, dct)
    display_xai(data, target, dct)
    display(Markdown("## Confusion Matrices:\n"))
    plot_confusion_matrices(data, target, predictions)


def prepare_df_for_modeling(df, only_important_columns=False):
    boolean_columns_to_keep = [
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
    'ocr_text_contains_they', 
    'transcript_contains_us', 
    'ocr_text_contains_us', 
    'transcript_contains_them', 
    'ocr_text_contains_them', 
    'transcript_contains_my', 
    'ocr_text_contains_my', 
    'transcript_contains_our', 
    'ocr_text_contains_our', 
    'transcript_contains_ours', 
    'ocr_text_contains_ours', 
    'transcript_contains_your', 
    'ocr_text_contains_your', 
    'transcript_contains_yours', 
    'ocr_text_contains_yours', 
    'transcript_contains_his', 
    'ocr_text_contains_his', 
    'transcript_contains_her',
    'ocr_text_contains_her',
    'transcript_contains_its',
    'ocr_text_contains_its',
    'transcript_contains_their',
    'ocr_text_contains_their',
    'transcript_contains_theirs',
    'ocr_text_contains_theirs',
    ]

    integer_columns_to_keep = [
        'transcript_superlative_count',
        'transcript_comparative_count',
        'transcript_uniqueness_count',
        'ocr_text_superlative_count',
        'ocr_text_comparative_count',
        'ocr_text_uniqueness_count',
        'ocr_text_total_bdm_terms_count',
        'transcript_num_adj_noun_pairs',
        'ocr_text_num_adj_noun_pairs',
        'transcript_num_comparisons',
        'ocr_text_num_comparisons',
    ]
    float_columns_to_keep = [
        'transcript_superlative_pct',
        'transcript_uniqueness_pct',
        'ocr_text_superlative_pct',
        'ocr_text_comparative_pct',
        'ocr_text_uniqueness_pct',
        'ocr_text_total_bdm_terms_pct',
        'transcript_product_cat_keywords_similarity',
        'ocr_text_product_cat_keywords_similarity',
        'transcript_product_brand_keywords_similarity',
        'ocr_text_product_brand_keywords_similarity',

    ]
    text_columns_to_keep = [
        'commercial_number',
    ]

    for col in integer_columns_to_keep:
        if col in df.columns:
            df[col] = df[col].astype(int)

    for col in float_columns_to_keep:
        if col in df.columns:
            df[col] = df[col].astype(float)
    for col in boolean_columns_to_keep:
        if col in df.columns:
            df[col] = df[col].astype(int)
    for col in text_columns_to_keep:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df = df.loc[:, (df.columns.isin(boolean_columns_to_keep) | df.columns.isin(integer_columns_to_keep) | df.columns.isin(float_columns_to_keep) | df.columns.isin(text_columns_to_keep))]

    important_columns = [
        'commercial_number',
        'transcript_product_cat_keywords_similarity', 
        'transcript_product_brand_keywords_similarity',
        'transcript_contains_you',
        'transcript_num_adj_noun_pairs',
        'transcript_contains_your',
        'transcript_contains_it',
        'csr_type',
        'transcript_total_bdm_terms_pct',
        'transcript_num_comparisons',
        'transcript_contains_we',
        'transcript_contains_us',
        'BDM'
        ]
    if only_important_columns:
        df = df.loc[:, df.columns.isin(important_columns)]
    # lastly sort the columns alphabetically
    df = df.reindex(sorted(df.columns), axis=1)
    return df

def display_decision_tree(data, target, model):
    """
    Display the decision tree visualization.
    
    Parameters:
    - data: Feature matrix (pandas DataFrame)
    - target: Target variable (pandas Series or numpy array)
    - model: Decision tree model instance
    """
    model.fit(data, target)  # Ensure the model is fitted

    # Create a new figure for the decision tree
    plt.figure(figsize=(20, 10))
    plot_tree(model, 
              filled=True, 
              feature_names=data.columns, 
              class_names=['No BDM', 'BDM'])
    plt.title("Decision Tree Visualization")
    plt.savefig("decision_tree_high_def.svg", format='svg')  # Save as SVG
    plt.show()
