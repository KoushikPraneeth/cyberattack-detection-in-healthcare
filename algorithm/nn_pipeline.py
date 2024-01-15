from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

class NNPipeline:
    def __init__(self):
        self.model = None

    async def preprocess_data(self, X_train, X_test):
        # Standardize the data using StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test

    async def build_model(self, input_dim, hidden_dim, output_dim):
        # Build a simple neural network model using Keras
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dense(hidden_dim, activation='relu'))
        model.add(Dense(output_dim, activation='sigmoid'))

        # Compile the model with binary cross-entropy loss and Adam optimizer
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    async def train_evaluate_model(self, X_train, X_test, y_train, y_test, epochs=10, batch_size=32):
        # Train the neural network model
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.2)

        # Make predictions on the test set
        y_pred = self.model.predict(X_test)

        # Convert probabilities to binary predictions
        y_pred = np.where(y_pred < 0.5, 0, 1)

        # Evaluate the model's performance using various metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("*****Neural Network Prediction Stats*****")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

    async def predict(self, features, target):
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Preprocess the data
        X_train, X_test = await self.preprocess_data(X_train, X_test)

        # Build and train the neural network model
        input_dim = X_train.shape[1]
        hidden_dim = 10
        output_dim = 1
        await self.build_model(input_dim, hidden_dim, output_dim)
        await self.train_evaluate_model(X_train, X_test, y_train, y_test, batch_size=100, epochs=20)
