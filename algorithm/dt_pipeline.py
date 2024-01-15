# Decision Tree Training and Evaluation Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

class DTPipeline:
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

        print("*****Decision Tree Best Model Prediction Stats*****")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

    async def predict(self, features, target):
        # Split the data and perform grid search for hyperparameter tuning
        data_splits = await self.split_data(features, target)
        X_train, X_test, y_train, y_test = data_splits

        # Define Decision Tree model and its hyperparameter grid
        dt_model = DecisionTreeClassifier()
        dt_params = {
            'max_depth': [10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['auto', 'sqrt']
        }

        # Perform GridSearchCV to find the best hyperparameters
        dt_grid_search = GridSearchCV(dt_model, dt_params, cv=5, scoring='accuracy', verbose=2)
        dt_grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_dt_params = dt_grid_search.best_params_
        print(f"Best params found for Decision Tree: {best_dt_params}")

        # Create the best Decision Tree model with the tuned hyperparameters
        best_dt_model = dt_grid_search.best_estimator_
        # Fit the model on the training data
        best_dt_model.fit(X_train, y_train)

        # Make predictions on the test data and evaluate the model
        y_pred = best_dt_model.predict(X_test)
        await self.evaluate(y_test, y_pred)
        return best_dt_model
