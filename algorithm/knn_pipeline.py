# K-Nearest Neighbor Training and Evaluation Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV


class KNNPipeline:
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

        print("*****K-Nearest Neighbors Best Model Prediction Stats*****")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

    async def predict(self, features, target):
        # Split the data and perform grid search for hyperparameter tuning
        data_splits = await self.split_data(features, target)
        X_train, X_test, y_train, y_test = data_splits

        # Define K-Nearest Neighbors model and its hyperparameter grid
        knn_model = KNeighborsClassifier()
        knn_params = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 40, 50],
            'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
        }

        # Perform GridSearchCV to find the best hyperparameters
        knn_grid_search = GridSearchCV(knn_model, knn_params, cv=5, scoring='accuracy', verbose=1)
        knn_grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_knn_params = knn_grid_search.best_params_
        print(f"Best params found for KNN: {best_knn_params}")

        # Create the best K-Nearest Neighbors model with the tuned hyperparameters
        best_knn_model = KNeighborsClassifier(
            n_neighbors=best_knn_params['n_neighbors'],
            weights=best_knn_params['weights'],
            algorithm=best_knn_params['algorithm'],
            leaf_size=best_knn_params['leaf_size'],
            p=best_knn_params['p']
        )

        # Train the model on the entire training dataset
        best_knn_model.fit(X_train, y_train)

        # Make predictions on the test data and evaluate the model
        y_pred = best_knn_model.predict(X_test)
        await self.evaluate(y_test, y_pred)
