import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
# for tf in venv
# import sys
# sys.path.append("c:/users/roman.khakhula/appdata/local/programs/python/python312/lib/site-packages")
import tensorflow as tf
import joblib
from pathlib import Path

def model_evaluation():
    # Завантаження даних
    data_path = Path('final_df.csv').absolute()
    data = pd.read_csv(data_path)
    
    # Розділення даних на ознаки та мітки
    X = data.drop('churn', axis=1)
    y = data['churn']

    # Розділення на тренувальний та тестовий набори
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Масштабування даних
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Завантаження моделей
    logistic_regression_model = joblib.load(Path('Logistic_Regression_model.pkl').absolute())
    decision_tree_model = joblib.load(Path('Decision_Tree_model.pkl').absolute())
    random_forest_model = joblib.load(Path('Random_Forest_model.pkl').absolute())
    neural_network_model = tf.keras.models.load_model(Path('neural_network_model.h5').absolute())

    models = {
        'Logistic Regression': logistic_regression_model,
        'Random Forest': random_forest_model,
        'Decision Tree': decision_tree_model,
        'Neural Network': neural_network_model
    }

    history = {}

    # Оцінка моделей
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        if model_name == 'Neural Network':
            y_pred = (model.predict(X_test) > 0.5).astype(int)
        else:
            y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        history[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'MSE' : mse
        }

        ''' print(f"{model_name} - Accuracy: {accuracy:.4f}")
        print(f"{model_name} - Precision: {precision:.4f}")
        print(f"{model_name} - Recall: {recall:.4f}")
        print(f"{model_name} - F1 Score: {f1:.4f}")
        print(f"{model_name} - MSE: {mse:.4f}")'''
    
    return history, scaler, models

# global scaler
# global models
model_history, scaler, models = model_evaluation()

if __name__ == "__main__":
    model_history = model_evaluation()
    print(model_history)