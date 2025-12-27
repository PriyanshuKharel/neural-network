import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


class DenseLayer:
    def __init__(self, input_size, output_size, activation='relu'):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        
        
        self.input = None
        self.z = None
        self.output = None
        
        
        self.dW = None
        self.db = None
    
    def forward(self, x):
        """Forward pass through the layer."""
        self.input = x
        self.z = np.dot(x, self.weights) + self.biases
        
        if self.activation == 'relu':
            self.output = relu(self.z)
        elif self.activation == 'sigmoid':
            self.output = sigmoid(self.z)
        elif self.activation == 'softmax':
            self.output = softmax(self.z)
        else:
            self.output = self.z  
        
        return self.output
    
    def backward(self, d_output, learning_rate):
        """Backward pass through the layer."""
        batch_size = self.input.shape[0]
        
        # Apply activation derivative
        if self.activation == 'relu':
            d_z = d_output * relu_derivative(self.z)
        elif self.activation == 'sigmoid':
            d_z = d_output * sigmoid_derivative(self.z)
        elif self.activation == 'softmax':
            # For softmax with cross-entropy, gradient is simplified
            d_z = d_output
        else:
            d_z = d_output
        
        # Compute gradients
        self.dW = np.dot(self.input.T, d_z) / batch_size
        self.db = np.mean(d_z, axis=0, keepdims=True)
        
        # Gradient to pass to previous layer
        d_input = np.dot(d_z, self.weights.T)
        
        # Update weights
        self.weights -= learning_rate * self.dW
        self.biases -= learning_rate * self.db
        
        return d_input



class ANN:
    """Artificial Neural Network."""
    
    def __init__(self):
        self.layers = []
    
    def add_layer(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)
    
    def forward(self, x):
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, y_pred, y_true, learning_rate):
        """Backward pass through all layers."""
        # Gradient of cross-entropy with softmax
        d_output = y_pred - y_true
        
        # Backpropagate through layers in reverse
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output, learning_rate)
    
    def train(self, X, y, epochs=100, learning_rate=0.01, batch_size=32, verbose=True):
        """Train the neural network."""
        n_samples = X.shape[0]
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                loss = cross_entropy_loss(y_pred, y_batch)
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass
                self.backward(y_pred, y_batch, learning_rate)
            
            # Calculate metrics
            avg_loss = epoch_loss / n_batches
            y_pred_all = self.forward(X)
            accuracy = np.mean(np.argmax(y_pred_all, axis=1) == np.argmax(y, axis=1))
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        
        return history
    
    def predict(self, X):
      
        return np.argmax(self.forward(X), axis=1)



def create_spiral_dataset(n_points=100, n_classes=3):
    
    np.random.seed(42)
    X = np.zeros((n_points * n_classes, 2))
    y = np.zeros(n_points * n_classes, dtype=int)
    
    for i in range(n_classes):
        idx = range(n_points * i, n_points * (i + 1))
        r = np.linspace(0.0, 1, n_points)  # radius
        t = np.linspace(i * 4, (i + 1) * 4, n_points) + np.random.randn(n_points) * 0.2
        X[idx] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[idx] = i
    
    return X, y


def one_hot_encode(y, n_classes):
    """One-hot encode labels."""
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def main():
    print("=" * 60)
    print("Artificial Neural Network (ANN) from Scratch")
    print("=" * 60)
    
    # Create dataset
    print("\nCreating spiral dataset...")
    X, y = create_spiral_dataset(n_points=100, n_classes=3)
    y_one_hot = one_hot_encode(y, n_classes=3)
    
    print(f"Dataset shape: X={X.shape}, y={y_one_hot.shape}")
    
    # Normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Create neural network
    print("\nANN:")
    print("   Input(2) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(3, Softmax)")
    
    model = ANN()
    model.add_layer(DenseLayer(2, 64, activation='relu'))
    model.add_layer(DenseLayer(64, 32, activation='relu'))
    model.add_layer(DenseLayer(32, 3, activation='softmax'))
    
    # Train
    print("\nTraining...")
    history = model.train(X, y_one_hot, epochs=100, learning_rate=0.1, batch_size=32)
    
    # Evaluate
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nFinal Training Accuracy: {accuracy:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot training loss
    axes[0].plot(history['loss'])
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Plot training accuracy
    axes[1].plot(history['accuracy'])
    axes[1].set_title('Training Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True, alpha=0.3)
    
    # Plot decision boundary
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axes[2].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    scatter = axes[2].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black', s=40)
    axes[2].set_title('Decision Boundary')
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('ann_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nResults saved to 'ann_results.png'")
    print("=" * 60)


if __name__ == "__main__":
    main()
