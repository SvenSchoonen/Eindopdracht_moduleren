# RLAagent.py
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class RLAagent:
    def __init__(self, grid):
        self.grid = grid
        self.data = []
        self.labels = []
        self.knn_model = None

    def collect_features(self):
        self.data = []
        self.labels = []
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                for z in range(self.grid.shape[2]):
                    cell = self.grid[x, y, z]
                    cell_features = [
                        x, y, z,  # Position
                        cell.cell_type,  # Type of cell
                        cell.mutation_count,  # Mutations
                        cell.state,  # State
                    ]
                    self.data.append(cell_features)
                    self.labels.append(cell.state)  # Predicting the next state

    def train_knn_model(self):
        self.collect_features()
        data = np.array(self.data)
        labels = np.array(self.labels)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        # Train KNN model
        self.knn_model = KNeighborsClassifier(n_neighbors=3)
        self.knn_model.fit(X_train, y_train)
        
        accuracy = self.knn_model.score(X_test, y_test)
        print(f"KNN Model accuracy: {accuracy}")

    def predict_next_states(self):
        if self.knn_model is None:
            raise ValueError("KNN model is not trained. Call `train_knn_model` first.")

        predictions = []
        for x in range(self.grid.shape[0]):
            for y in range(self.grid.shape[1]):
                for z in range(self.grid.shape[2]):
                    cell = self.grid[x, y, z]
                    cell_features = np.array([[
                        x, y, z,
                        cell.cell_type,
                        cell.mutation_count,
                        cell.state,
                    ]])
                    prediction = self.knn_model.predict(cell_features)
                    predictions.append((cell.position, prediction[0]))
        return predictions
