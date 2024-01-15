# Random Forest Training and Evaluation Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV


class RFPipeline:
    @staticmethod
    async def split_data(features, target):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    async def preprocess_data(self, data, drop_features, target_feature):
        # Preprocess the data by dropping features, encoding categorical variables, and handling missing values
        df = data.drop(drop_features, axis=1)
        df[target_feature].fillna(0, inplace=True)

        le = LabelEncoder()
        df['Protocol'] = le.fit_transform(df['Protocol'])
        df['Packet Type'] = le.fit_transform(df['Packet Type'])
        df['Traffic Type'] = le.fit_transform(df['Traffic Type'])
        df['Severity Level'] = le.fit_transform(df['Severity Level'])

        df['Malware Indicators'] = df['Malware Indicators'].apply(lambda x: 1 if x == 'IoC Detected' else x)

        X = df.drop(columns=[target_feature])
        y = df[target_feature]

        # Standardize the features using StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return X, y

    @staticmethod
    async def evaluate(actual, pred):
        # Evaluate the model's performance using various metrics
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        f1 = f1_score(actual, pred)

        print("*****Random Forest Prediction Stats*****")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

    async def predict(self, features, target):
        # Split the dataset into training and testing sets
        data_splits = await self.split_data(features, target)
        X_train, X_test, y_train, y_test = data_splits

        # Initialize and train the Random Forest model
        rf_model = RandomForestClassifier()
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['log2', 'sqrt']
        }
        rf_grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', verbose=2)
        rf_grid_search.fit(X_train, y_train)
        best_model = rf_grid_search.best_estimator_
        best_model.fit(X_train, y_train)

        # Make predictions on the test set and evaluate the model
        y_pred = best_model.predict(X_test)
        await self.evaluate(y_test, y_pred)
        return best_model
