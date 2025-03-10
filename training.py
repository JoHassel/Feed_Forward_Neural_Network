import architecture
# import utils 
import numpy as np
from utils import get_mnist
from typing import List, Any

# Define paramater:
learning_rates = [0.01, 0.1, 1.0]
batch_size = 60
num_epochs = 20

# Global lists for stats:
histories = []
test_accuracies = []


def get_data() -> tuple:
    X, y = get_mnist()

    start_train_idx = 0
    end_train_idx = 50000
    start_val_idx = end_train_idx
    end_val_idx = 60000
    start_test_idx = end_val_idx
    end_test_idx = 70000

    X_train, y_train = X[start_train_idx:end_train_idx], y[start_train_idx:end_train_idx]
    X_val, y_val = X[start_val_idx:end_val_idx], y[start_val_idx:end_val_idx]
    X_test, y_test = X[start_test_idx:end_test_idx], y[start_test_idx:end_test_idx]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_model():
    X_train, y_train, X_val, y_val, X_test, y_test = get_data()
    input_dim = X_train.shape[1] # 28x28 pixels
    output_dim = 10 # Digits 0-9

    for lr in learning_rates:
        print(f"\nTraining model with learning rate {lr}...")

        layers: List[Any] = [
            architecture.Linear(input_dim, 64),
            architecture.ReLU(),
            architecture.Linear(64, output_dim)
        ]

        loss_fn = architecture.CrossEntropy()

        model = architecture.Model(layers)
        
        # Train the model
        history = model.train(
            loss_fn=loss_fn,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=lr
        )
        histories.append((lr, history))

        # Evaluate the model on the test set
        ### BEGIN SOLUTION
        test_logits = model.forward(X_test, training=False)
        test_predictions = np.argmax(test_logits, axis=1)
        test_accuracy = np.mean(test_predictions == y_test)
        test_accuracies.append((lr, test_accuracy))
        ### END
        print(f"Test Accuracy for learning rate {lr}: {test_accuracy * 100:.2f}%")

def plot(hist: list, acc: list):
    # Plotting Code
    import matplotlib.pyplot as plt

    # Plot validation accuracy across epochs for each model
    plt.figure(figsize=(10, 6))
    colors = []

    for lr, history in histories:
        val_acc = history['val_acc']
        line, = plt.plot(range(1, num_epochs + 1), val_acc, label=f'LR = {lr}')
        colors.append(line.get_color())  # Store the color used

    # Plot test accuracy as horizontal lines with the same color
    for idx, (lr, test_accuracy) in enumerate(test_accuracies):
        plt.hlines(
            test_accuracy,
            1,
            num_epochs,
            linestyles='dashed',
            colors=colors[idx],
            label=f'Test Acc LR={lr}'
        )

    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    ### END SOLUTION

    plt.grid(True)
    plt.show()

train_model()
plot(histories, test_accuracies)