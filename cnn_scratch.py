import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))




class Conv2D:    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weights = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.biases = np.zeros(out_channels)
        
        
        self.input = None
        self.output = None
    
    def forward(self, x):
        """Forward pass - perform convolution."""
        self.input = x
        batch_size, in_channels, in_height, in_width = x.shape
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Apply padding
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), 
                                  (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x
        
        
        self.output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Perform convolution
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        receptive_field = x_padded[b, :, 
                                                    h_start:h_start + self.kernel_size,
                                                    w_start:w_start + self.kernel_size]
                        self.output[b, oc, h, w] = np.sum(
                            receptive_field * self.weights[oc]
                        ) + self.biases[oc]
        
        return self.output
    
    def backward(self, d_output, learning_rate):
        """Backward pass through conv layer."""
        batch_size, _, out_height, out_width = d_output.shape
        
        # Initialize gradients
        dW = np.zeros_like(self.weights)
        db = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.input)
        
        # Apply padding to input
        if self.padding > 0:
            input_padded = np.pad(self.input, ((0, 0), (0, 0), 
                                               (self.padding, self.padding), 
                                               (self.padding, self.padding)), mode='constant')
            d_input_padded = np.zeros_like(input_padded)
        else:
            input_padded = self.input
            d_input_padded = np.zeros_like(self.input)
        
        # Compute gradients
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        
                        receptive_field = input_padded[b, :,
                                                        h_start:h_start + self.kernel_size,
                                                        w_start:w_start + self.kernel_size]
                        
                        dW[oc] += d_output[b, oc, h, w] * receptive_field
                        db[oc] += d_output[b, oc, h, w]
                        
                        d_input_padded[b, :,
                                       h_start:h_start + self.kernel_size,
                                       w_start:w_start + self.kernel_size] += \
                            d_output[b, oc, h, w] * self.weights[oc]
        
        # Remove padding from gradient
        if self.padding > 0:
            d_input = d_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_input = d_input_padded
        
        # Average gradients over batch
        dW /= batch_size
        db /= batch_size
        
        # Update weights
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db
        
        return d_input


class MaxPool2D:
    """Max pooling layer."""
    
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.max_indices = None
    
    def forward(self, x):
        """Forward pass - max pooling."""
        self.input = x
        batch_size, channels, in_height, in_width = x.shape
        
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.max_indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=int)
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        pool_region = x[b, c, 
                                        h_start:h_start + self.pool_size,
                                        w_start:w_start + self.pool_size]
                        
                        max_val = np.max(pool_region)
                        output[b, c, h, w] = max_val
                        
                        # Store index of max value for backprop
                        max_idx = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                        self.max_indices[b, c, h, w] = [h_start + max_idx[0], w_start + max_idx[1]]
        
        return output
    
    def backward(self, d_output, learning_rate):
        """Backward pass through max pooling."""
        d_input = np.zeros_like(self.input)
        batch_size, channels, out_height, out_width = d_output.shape
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        max_h, max_w = self.max_indices[b, c, h, w]
                        d_input[b, c, max_h, max_w] += d_output[b, c, h, w]
        
        return d_input


# ============================================================================
# Flatten Layer
# ============================================================================

class Flatten:
    """Flatten layer to convert 4D tensor to 2D."""
    
    def __init__(self):
        self.input_shape = None
    
    def forward(self, x):
        """Flatten the input."""
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, d_output, learning_rate):
        """Reshape gradient back to original shape."""
        return d_output.reshape(self.input_shape)


# ============================================================================
# Dense Layer
# ============================================================================

class Dense:
    """Fully connected layer."""
    
    def __init__(self, input_size, output_size, activation='relu'):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.activation = activation
        self.input = None
        self.z = None
    
    def forward(self, x):
        """Forward pass."""
        self.input = x
        self.z = np.dot(x, self.weights) + self.biases
        
        if self.activation == 'relu':
            return relu(self.z)
        elif self.activation == 'softmax':
            return softmax(self.z)
        return self.z
    
    def backward(self, d_output, learning_rate):
        """Backward pass."""
        batch_size = self.input.shape[0]
        
        if self.activation == 'relu':
            d_z = d_output * relu_derivative(self.z)
        else:
            d_z = d_output
        
        dW = np.dot(self.input.T, d_z) / batch_size
        db = np.mean(d_z, axis=0, keepdims=True)
        d_input = np.dot(d_z, self.weights.T)
        
        self.weights -= learning_rate * dW
        self.biases -= learning_rate * db
        
        return d_input


# ============================================================================
# ReLU Activation Layer
# ============================================================================

class ReLU:
    """ReLU activation layer."""
    
    def __init__(self):
        self.input = None
    
    def forward(self, x):
        self.input = x
        return relu(x)
    
    def backward(self, d_output, learning_rate):
        return d_output * relu_derivative(self.input)


class CNN:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, y_pred, y_true, learning_rate):
        d_output = y_pred - y_true
        
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output, learning_rate)
    
    def train(self, X, y, epochs=50, learning_rate=0.01, batch_size=16, verbose=True):
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
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
        
        return history
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


def create_synthetic_digits(n_samples=500, img_size=8, n_classes=4):
    """
    Create a synthetic dataset of simple patterns.
    Classes: horizontal lines, vertical lines, diagonal, cross patterns
    """
    np.random.seed(42)
    X = np.zeros((n_samples, 1, img_size, img_size))
    y = np.zeros(n_samples, dtype=int)
    
    samples_per_class = n_samples // n_classes
    
    for i in range(n_samples):
        class_idx = i // samples_per_class
        if class_idx >= n_classes:
            class_idx = n_classes - 1
        y[i] = class_idx
        
        if class_idx == 0:  # Horizontal lines
            row = np.random.randint(1, img_size - 1)
            X[i, 0, row, :] = 1.0
            X[i, 0, row-1:row+2, :] += np.random.randn(3, img_size) * 0.1
        
        elif class_idx == 1:  # Vertical lines
            col = np.random.randint(1, img_size - 1)
            X[i, 0, :, col] = 1.0
            X[i, 0, :, col-1:col+2] += np.random.randn(img_size, 3) * 0.1
        
        elif class_idx == 2:  # Diagonal
            for k in range(img_size):
                X[i, 0, k, k] = 1.0
            X[i, 0] += np.random.randn(img_size, img_size) * 0.1
        
        else:  # Cross pattern
            mid = img_size // 2
            X[i, 0, mid, :] = 1.0
            X[i, 0, :, mid] = 1.0
            X[i, 0] += np.random.randn(img_size, img_size) * 0.1
    
    # Normalize
    X = np.clip(X, 0, 1)
    
    return X, y


def one_hot_encode(y, n_classes):
    """One-hot encode labels."""
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot


def main():
    print("=" * 60)
    print("Convolutional Neural Network (CNN) from Scratch")
    print("=" * 60)
    
    # Create dataset
    print("\nCreating synthetic image dataset...")
    X, y = create_synthetic_digits(n_samples=400, img_size=8, n_classes=4)
    y_one_hot = one_hot_encode(y, n_classes=4)
    
    print(f" Dataset shape: X={X.shape}, y={y_one_hot.shape}")
    print(" Classes: Horizontal, Vertical, Diagonal, Cross patterns")
    
    
    print("\nBuilding CNN architecture:")
    print("   Input(1x8x8) -> Conv2D(8, 3x3) -> ReLU -> MaxPool(2x2)")
    print("   -> Flatten -> Dense(64) -> Dense(4, Softmax)")
    
    model = CNN()
    model.add(Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding=1))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=2, stride=2))
    model.add(Flatten())
    model.add(Dense(8 * 4 * 4, 64, activation='relu'))
    model.add(Dense(64, 4, activation='softmax'))
    
    # Train
    print("\nTraining...")
    history = model.train(X, y_one_hot, epochs=30, learning_rate=0.05, batch_size=16)
    
    # Evaluate
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\nFinal Training Accuracy: {accuracy:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # Training metrics
    axes[0, 0].plot(history['loss'])
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(history['accuracy'])
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Sample images from each class
    class_names = ['Horizontal', 'Vertical', 'Diagonal', 'Cross']
    for idx, class_idx in enumerate(range(4)):
        sample_idx = np.where(y == class_idx)[0][0]
        ax = axes[1, idx] if idx < 3 else axes[0, 2]
        ax.imshow(X[sample_idx, 0], cmap='gray')
        pred = predictions[sample_idx]
        ax.set_title(f'True: {class_names[class_idx]}\nPred: {class_names[pred]}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('cnn_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nResults saved to 'cnn_results.png'")
    print("=" * 60)


if __name__ == "__main__":
    main()
