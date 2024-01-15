# XGBoost Training and Evaluation Pipeline
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV


class XGBPipeline:
    @staticmethod
    async def split_data(features, target):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    @staticmethod
    async def evaluate(actual, pred):
        # Evaluate the model's performance using various metrics
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)

        print("*****XGBoost Prediction Stats*****")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

    async def predict(self, features, target):
        # Split the dataset into training and testing sets
        data_splits = await self.split_data(features, target)
        X_train, X_test, y_train, y_test = data_splits

        # Initialize and train the XGBoost model
        xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5],
            'min_child_weight': [1, 3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        xgb_grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', verbose=2)
        xgb_grid_search.fit(X_train, y_train)
        best_model = xgb_grid_search.best_estimator_

        best_model.fit(X_train, y_train)

        # Make predictions on the test set and evaluate the model
        y_pred = best_model.predict(X_test)
        await self.evaluate(y_test, y_pred)
        return best_model
