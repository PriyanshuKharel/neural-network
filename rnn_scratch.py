import numpy as np
import matplotlib.pyplot as plt


def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy_loss(y_pred, y_true):
    """Cross-entropy loss for classification."""
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))


class RNNCell:  
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        
        self.W_xh = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / (input_size + hidden_size))
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / (hidden_size * 2))
        self.b_h = np.zeros((1, hidden_size))
        
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_h = np.zeros_like(self.b_h)
        
        self.cache = []
    
    def forward(self, x_sequence, h_init=None):
        batch_size, seq_length, _ = x_sequence.shape
        
        if h_init is None:
            h_prev = np.zeros((batch_size, self.hidden_size))
        else:
            h_prev = h_init
        
        self.cache = []
        h_all = []
        
        for t in range(seq_length):
            x_t = x_sequence[:, t, :]
            
            # RNN computation
            z = np.dot(x_t, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h
            h_t = tanh(z)
            
            # Store for backprop
            self.cache.append((x_t, h_prev, z, h_t))
            
            h_all.append(h_t)
            h_prev = h_t
        
        h_all = np.stack(h_all, axis=1)  # (batch_size, seq_length, hidden_size)
        h_final = h_all[:, -1, :]
        
        return h_all, h_final
    
    def backward(self, dh_all, dh_final=None):
        batch_size = dh_all.shape[0]
        seq_length = len(self.cache)
        
        # Reset gradients
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_h = np.zeros_like(self.b_h)
        
        dx_sequence = np.zeros((batch_size, seq_length, self.input_size))
        
        # Initialize dh_next
        if dh_final is not None:
            dh_next = dh_final.copy()
        else:
            dh_next = np.zeros((batch_size, self.hidden_size))
        
        # Backprop through time
        for t in reversed(range(seq_length)):
            x_t, h_prev, z, h_t = self.cache[t]
            
            # Add gradient from output at this timestep and from next timestep
            dh = dh_all[:, t, :] + dh_next
            
            # Gradient through tanh
            dz = dh * tanh_derivative(z)
            
            # Gradients for weights
            self.dW_xh += np.dot(x_t.T, dz)
            self.dW_hh += np.dot(h_prev.T, dz)
            self.db_h += np.sum(dz, axis=0, keepdims=True)
            
            # Gradient for input
            dx_sequence[:, t, :] = np.dot(dz, self.W_xh.T)
            
            # Gradient for previous hidden state
            dh_next = np.dot(dz, self.W_hh.T)
        
        # Average gradients over batch
        self.dW_xh /= batch_size
        self.dW_hh /= batch_size
        self.db_h /= batch_size
        
        # Gradient clipping to prevent exploding gradients
        clip_value = 5.0
        self.dW_xh = np.clip(self.dW_xh, -clip_value, clip_value)
        self.dW_hh = np.clip(self.dW_hh, -clip_value, clip_value)
        self.db_h = np.clip(self.db_h, -clip_value, clip_value)
        
        return dx_sequence
    
    def update_weights(self, learning_rate):
        """Update weights using gradients."""
        self.W_xh -= learning_rate * self.dW_xh
        self.W_hh -= learning_rate * self.dW_hh
        self.b_h -= learning_rate * self.db_h


class OutputLayer:
    """Dense output layer for classification."""
    
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros((1, output_size))
        self.input = None
    
    def forward(self, x):
        """Forward pass with softmax."""
        self.input = x
        z = np.dot(x, self.W) + self.b
        return softmax(z)
    
    def backward(self, y_pred, y_true, learning_rate):
        """Backward pass."""
        batch_size = y_pred.shape[0]
        
        # Gradient of cross-entropy + softmax
        dz = y_pred - y_true
        
        # Gradients
        dW = np.dot(self.input.T, dz) / batch_size
        db = np.mean(dz, axis=0, keepdims=True)
        d_input = np.dot(dz, self.W.T)
        
        # Update weights
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        return d_input


class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.rnn_cell = RNNCell(input_size, hidden_size)
        self.output_layer = OutputLayer(hidden_size, output_size)
    
    def forward(self, x_sequence):
        """Forward pass."""
        h_all, h_final = self.rnn_cell.forward(x_sequence)
        y_pred = self.output_layer.forward(h_final)
        return y_pred, h_all, h_final
    
    def backward(self, y_pred, y_true, learning_rate):
        """Backward pass."""
        # Backprop through output layer
        dh_final = self.output_layer.backward(y_pred, y_true, learning_rate)
        
        
        batch_size = y_pred.shape[0]
        seq_length = len(self.rnn_cell.cache)
        dh_all = np.zeros((batch_size, seq_length, self.rnn_cell.hidden_size))
        
        # Backprop through RNN
        self.rnn_cell.backward(dh_all, dh_final)
        self.rnn_cell.update_weights(learning_rate)
    
    def train(self, X, y, epochs=100, learning_rate=0.01, batch_size=32, verbose=True):
        """Train the RNN."""
        n_samples = X.shape[0]
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred, _, _ = self.forward(X_batch)
                
                # Compute loss
                loss = cross_entropy_loss(y_pred, y_batch)
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass
                self.backward(y_pred, y_batch, learning_rate)
            
            # Calculate metrics
            avg_loss = epoch_loss / n_batches
            y_pred_all, _, _ = self.forward(X)
            accuracy = np.mean(np.argmax(y_pred_all, axis=1) == np.argmax(y, axis=1))
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        
        return history
    
    def predict(self, X):
        """Make predictions."""
        y_pred, _, _ = self.forward(X)
        return np.argmax(y_pred, axis=1)



def create_sequence_dataset(n_samples=500, seq_length=10, n_classes=3):
    np.random.seed(42)
    X = np.zeros((n_samples, seq_length, 1))
    y = np.zeros(n_samples, dtype=int)
    
    samples_per_class = n_samples // n_classes
    
    for i in range(n_samples):
        class_idx = min(i // samples_per_class, n_classes - 1)
        y[i] = class_idx
        
        if class_idx == 0:  # Increasing
            base = np.linspace(0, 1, seq_length)
            noise = np.random.randn(seq_length) * 0.1
            X[i, :, 0] = base + noise
        
        elif class_idx == 1:  # Decreasing
            base = np.linspace(1, 0, seq_length)
            noise = np.random.randn(seq_length) * 0.1
            X[i, :, 0] = base + noise
        
        else:  # Oscillating
            t = np.linspace(0, 4 * np.pi, seq_length)
            base = np.sin(t) * 0.5 + 0.5
            noise = np.random.randn(seq_length) * 0.1
            X[i, :, 0] = base + noise
    
    # Normalize
    X = (X - X.mean()) / (X.std() + 1e-8)
    
    return X, y


def one_hot_encode(y, n_classes):
    """One-hot encode labels."""
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def plot_sample_sequences(X, y, class_names, n_samples=3):
    """Plot sample sequences from each class."""
    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(12, 3 * n_classes))
    
    for class_idx in range(n_classes):
        class_samples = np.where(y == class_idx)[0][:n_samples]
        for j, sample_idx in enumerate(class_samples):
            ax = axes[class_idx, j] if n_classes > 1 else axes[j]
            ax.plot(X[sample_idx, :, 0], marker='o', markersize=3)
            ax.set_title(f'{class_names[class_idx]} (Sample {j+1})')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    print("=" * 60)
    print("Recurrent Neural Network (RNN) from Scratch")
    print("=" * 60)
    
    # Create dataset
    print("\nCreating synthetic sequence dataset...")
    seq_length = 15
    n_classes = 3
    X, y = create_sequence_dataset(n_samples=600, seq_length=seq_length, n_classes=n_classes)
    y_one_hot = one_hot_encode(y, n_classes=n_classes)
    
    print(f"Dataset shape: X={X.shape}, y={y_one_hot.shape}")
    print("Classes: Increasing, Decreasing, Oscillating patterns")
    
    # Shuffle and split into train/test
    indices = np.random.permutation(len(X))
    X_shuffled, y_shuffled = X[indices], y[indices]
    y_one_hot_shuffled = y_one_hot[indices]
    
    n_train = int(0.8 * len(X))
    X_train, X_test = X_shuffled[:n_train], X_shuffled[n_train:]
    y_train, y_test = y_one_hot_shuffled[:n_train], y_one_hot_shuffled[n_train:]
    y_test_labels = y_shuffled[n_train:]
    
    # Create RNN
    print("\nBuilding RNN architecture:")
    print(f"  Input(seq_length={seq_length}, features=1) -> RNN(hidden=32) -> Dense({n_classes}, Softmax)")
    
    model = RNN(input_size=1, hidden_size=32, output_size=n_classes)
    
    # Train
    print("\nTraining...")
    history = model.train(X_train, y_train, epochs=100, learning_rate=0.05, batch_size=32)
    
    # Evaluate
    train_predictions = model.predict(X_train)
    train_accuracy = np.mean(train_predictions == np.argmax(y_train, axis=1))
    
    test_predictions = model.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test_labels)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    
    # Plot results
    class_names = ['Increasing', 'Decreasing', 'Oscillating']
    
    fig = plt.figure(figsize=(15, 10))
    
    # Training loss
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(history['loss'])
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Training accuracy
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(history['accuracy'])
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    
    # Sample predictions
    for idx, class_idx in enumerate(range(n_classes)):
        sample_idx = np.where(y_test_labels == class_idx)[0][0]
        ax = fig.add_subplot(2, 3, idx + 4)
        ax.plot(X_test[sample_idx, :, 0], marker='o', markersize=4, linewidth=2)
        pred = test_predictions[sample_idx]
        color = 'green' if pred == class_idx else 'red'
        ax.set_title(f'True: {class_names[class_idx]}\nPred: {class_names[pred]}', color=color)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rnn_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nResults saved to 'rnn_results.png'")
    print("=" * 60)


if __name__ == "__main__":
    main()
