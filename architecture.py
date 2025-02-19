import numpy as np
from typing import Any, List, Optional


class Linear:
    """Implements a linear (fully connected) layer."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """
        Initialize the layer with input and output dimensions.

        Args:
            input_dim (int): The size of each input sample.
            output_dim (int): The size of each output sample.
        """
        self.weights: np.ndarray = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.biases: np.ndarray = np.zeros((1, output_dim))
        self.grad_weights: Optional[np.ndarray] = None
        self.grad_biases: Optional[np.ndarray] = None
        self.input: Optional[np.ndarray] = None
        

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Make the layer callable.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Output of the linear layer.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of the linear layer.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: The linear transformation of the input.
        """
        self.input = x
        output: np.ndarray = x @ self.weights + self.biases
        return output
        

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass of the linear layer.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        self.grad_weights = self.input.T @ grad_output
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        grad_input: np.ndarray = grad_output @ self.weights.T
        return grad_input
        
        
    def update_params(self, learning_rate: float) -> None:
        """
        Update the weights and biases using the computed gradients.

        Args:
            learning_rate (float): The learning rate for parameter updates.
        """
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases
        


class ReLU:
    """Implements the ReLU activation function."""

    def __init__(self) -> None:
        """Initialize the ReLU activation function."""
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Make the ReLU activation callable.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Output after applying ReLU.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of ReLU activation.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Output after applying ReLU.
        """
        self.input = x
        output: np.ndarray = np.maximum(0, x)
        return output
        
        
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass of ReLU activation.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        grad_input: np.ndarray = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input
        


class CrossEntropy:
    """Implements the cross-entropy loss function with softmax."""

    def __init__(self) -> None:
        """Initialize the CrossEntropy loss function."""
        self.logits: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.probs: Optional[np.ndarray] = None
        
        
    def __call__(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the cross-entropy loss.

        Args:
            logits (np.ndarray): Logits predicted by the model.
            labels (np.ndarray): True labels.

        Returns:
            float: The cross-entropy loss.
        """
        return self.forward(logits, labels)


    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the forward pass of the loss function.

        Args:
            logits (np.ndarray): Logits predicted by the model.
            labels (np.ndarray): True labels.

        Returns:
            float: The cross-entropy loss.
        """        
        self.logits = logits
        self.labels = labels
        exp_logits: np.ndarray = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        N: int = logits.shape[0]
        correct_logprobs: np.ndarray = -np.log(self.probs[range(N), labels])
        loss: float = np.sum(correct_logprobs) / N
        return loss
        
        
    def backward(self) -> np.ndarray:
        """
        Compute the backward pass of the loss function.

        Returns:
            np.ndarray: Gradient of the loss with respect to the logits.
        """
        N: int = self.logits.shape[0]
        grad_logits: np.ndarray = self.probs.copy()
        grad_logits[range(N), self.labels] -= 1
        grad_logits /= N
        return grad_logits



class Model:
    """Represents a feed-forward neural network model."""

    def __init__(self, layers: List[Any]) -> None:
        """
        Initialize the model with a list of layers.

        Args:
            layers (List[Any]): List of layers (e.g., Linear, ReLU).
        """        
        self.layers: List[Any] = layers
        
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Perform a forward pass through all layers.

        Args:
            x (np.ndarray): Input data.
            training (bool, optional): Whether in training mode. Defaults to True.

        Returns:
            np.ndarray: Output of the model.
        """     
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
        
        
    def backward(self, grad_output: np.ndarray) -> None:
        """
        Perform a backward pass through all layers.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output.
        """
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        
        
    def update_params(self, learning_rate: float) -> None:
        """
        Update parameters of all layers.

        Args:
            learning_rate (float): Learning rate for parameter updates.
        """
        for layer in self.layers:
            if hasattr(layer, 'update_params'):
                layer.update_params(learning_rate)
        
        
    def train(self, loss_fn: CrossEntropy,
              X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              batch_size: int, num_epochs: int, learning_rate: float) -> dict:
        """
        Train the model using stochastic gradient descent.

        Args:
            loss_fn (CrossEntropy): The loss function to use.
            X_train (np.ndarray): Training data inputs.
            y_train (np.ndarray): Training data labels.
            X_val (np.ndarray): Validation data inputs.
            y_val (np.ndarray): Validation data labels.
            batch_size (int): Size of each training batch.
            num_epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for parameter updates.

        Returns:
            dict: Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        """
        num_samples: int = X_train.shape[0]
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        for epoch in range(num_epochs):
            indices: np.ndarray = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train_shuffled: np.ndarray = X_train[indices]
            y_train_shuffled: np.ndarray = y_train[indices]
            
            epoch_loss = 0
            correct_preds = 0
            num_batches = 0

            for start_idx in range(0, num_samples, batch_size):
                end_idx: int = start_idx + batch_size
                X_batch: np.ndarray = X_train_shuffled[start_idx:end_idx]
                y_batch: np.ndarray = y_train_shuffled[start_idx:end_idx]

                # Forward pass
                out: np.ndarray = self.forward(X_batch, training=True)

                # Compute loss
                loss: float = loss_fn(out, y_batch)
                epoch_loss += loss * X_batch.shape[0]

                # Compute training accuracy
                preds = np.argmax(out, axis=1)
                correct_preds += np.sum(preds == y_batch)

                # Backward pass
                grad_output: np.ndarray = loss_fn.backward()
                self.backward(grad_output)

                # Update parameters
                self.update_params(learning_rate)

                num_batches += 1

            # Compute average loss and accuracy over the epoch
            avg_train_loss = epoch_loss / num_samples
            train_accuracy = correct_preds / num_samples

            # Validation
            val_out: np.ndarray = self.forward(X_val, training=False)
            val_loss: float = loss_fn(val_out, y_val)
            val_preds = np.argmax(val_out, axis=1)
            val_accuracy = np.mean(val_preds == y_val)

            # Record history
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_accuracy)

            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Train Loss: {avg_train_loss:.6f}, Train Acc: {train_accuracy * 100:.2f}%, "
                  f"Val Loss: {val_loss:.6f}, Val Acc: {val_accuracy * 100:.2f}%")

        return history
    
