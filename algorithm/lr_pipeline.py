# Logistic Regression Training and Evaluation Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

class LRPipeline:
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

        print("*****Logistic Regression Best Model Prediction Stats*****")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

    async def predict(self, features, target):
        # Split the data and perform grid search for hyperparameter tuning
        data_splits = await self.split_data(features, target)
        X_train, X_test, y_train, y_test = data_splits

        # Define Logistic Regression model and its hyperparameter grid
        lr_model = LogisticRegression()
        lr_params = {
            'penalty': ['l1', 'l2', 'elasticnet'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'l1_ratio': [0.5]
        }

        # Perform GridSearchCV to find the best hyperparameters
        lr_grid_search = GridSearchCV(lr_model, lr_params, cv=5, scoring='accuracy', verbose=1)
        lr_grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_lr_params = lr_grid_search.best_params_
        print(f"Best params found for Logistic Regression: {best_lr_params}")

        # Create the best Logistic Regression model with the tuned hyperparameters
        best_lr_model = LogisticRegression(
            penalty=best_lr_params['penalty'],
            C=best_lr_params['C'],
            solver=best_lr_params['solver'],
            random_state=42,
            max_iter=2000
        )

        # Train the model on the entire training dataset
        best_lr_model.fit(X_train, y_train)

        # Make predictions on the test data and evaluate the model
        y_pred = best_lr_model.predict(X_test)
        await self.evaluate(y_test, y_pred)
