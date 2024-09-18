import numpy as np

class FeedForwardNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward(self, X):
        # Forward propagation
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)
        
        return self.final_output
    
    def compute_loss(self, y_true, y_pred):
        # Mean squared error loss
        return np.mean((y_true - y_pred) ** 2)
    
    def backward(self, X, y_true, y_pred):
        # Backward propagation
        output_error = y_pred - y_true
        output_delta = output_error * self.sigmoid_derivative(y_pred)
        
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output -= np.dot(self.hidden_output.T, output_delta) * self.learning_rate
        self.bias_output -= np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        
        self.weights_input_hidden -= np.dot(X.T, hidden_delta) * self.learning_rate
        self.bias_hidden -= np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y, y_pred)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict(self, X):
        return self.forward(X)

# Function to get user input
def get_user_input():
    num_samples = int(input("Enter the number of training samples: "))
    input_size = int(input("Enter the number of input features: "))
    
    X = []
    print("Enter the input features (one sample per line, space-separated):")
    for _ in range(num_samples):
        sample = list(map(float, input().split()))
        X.append(sample)

    y = []
    print("Enter the corresponding labels (one label per line):")
    for _ in range(num_samples):
        label = float(input())
        y.append(label)

    epochs = int(input("Enter the number of training epochs: "))
    hidden_size = int(input("Enter the number of hidden neurons: "))
    learning_rate = float(input("Enter the learning rate: "))

    return np.array(X), np.array(y).reshape(-1, 1), epochs, hidden_size, learning_rate

# Main execution
if __name__ == "__main__":
    X, y, epochs, hidden_size, learning_rate = get_user_input()
    
    # Initialize the neural network
    input_size = X.shape[1]
    output_size = 1  # For binary classification
    nn = FeedForwardNeuralNetwork(input_size, hidden_size, output_size, learning_rate)

    # Train the neural network
    nn.train(X, y, epochs)

    # Make predictions
    predictions = nn.predict(X)
    print("Predictions after training:")
    print(predictions)
