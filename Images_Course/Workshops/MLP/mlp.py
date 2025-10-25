import numpy as np

class MLP:
    def __init__(self, data, hidden_dim, output=1, lr = 0.01):
        np.random.seed(42)
        self.lr = lr
        self.has_hidden = hidden_dim is not None

        if self.has_hidden: #hidden layer
            self.W_hidden = np.random.randn(data, hidden_dim)
            self.b_hidden = np.zeros((1, hidden_dim))
            self.W_out = np.random.randn(hidden_dim, output)
            self.b_out = np.zeros((1, output))
        else: #simple
            self.W = np.random.randn(data, output)
            self.b = np.zeros((1, output))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _dx_sigmoid(self, x):
        return x * (1-x)
    
    def _step(self, x):
        return np.where(x >= 0, 1, 0)
    
    def forward(self, X):
        """
        X: Input
        h_inp: input hidden layer
        h_out: output hidden layer
        out_inp: input output layer
        out_pred: output (prediction) output layer
        """
        if self.has_hidden:
            h_inp = np.dot(X, self.W_hidden) + self.b_hidden
            h_out = self._sigmoid(h_inp)

            out_inp = np.dot(h_out, self.W_out) + self.b_out
            out_pred = self._sigmoid(out_inp)

            self.cache = (X, h_out, out_pred)
            return out_pred
        else:
            z = np.dot(X, self.W) + self.b
            return self._step(z)


    def backward(self, X, y):
        """
        X: Input
        y: Target
        grad_out: gradient output layer
        grad_hidden: gradient hidden layer
        """
        if self.has_hidden:
            _, h_out, out_pred = self.cache

            #ouput layer
            error = y - out_pred
            grand_out = error * self._dx_sigmoid(out_pred)

            #hidden layer
            grand_hidden = np.dot(grand_out, self.W_out.T) * self._dx_sigmoid(h_out)

            #update weights
            self.W_out += self.lr * np.dot(h_out.T, grand_out)
            self.b_out += self.lr * np.sum(grand_out, axis=0, keepdims=True)
            self.W_hidden += self.lr * np.dot(X.T, grand_hidden)
            self.b_hidden += self.lr * np.sum(grand_hidden, axis=0, keepdims=True)
        else:
            for xi, target_val in zip(X, y):
                z = np.dot(xi, self.W) + self.b
                y_pred = self._step(z)
                error = target_val - y_pred
                self.W += self.lr * error * xi.reshape(-1, 1)
                self.b += self.lr * error

    def train(self, X, y, epochs = 100, show_err = False, each = 5):
        for i in range(epochs):
            out = self.forward(X)
            self.backward(X, y)
            if show_err and i % each == 0 and self.has_hidden:
                loss = np.mean((y - out) ** 2)
                print(f"Epoch {i}, Loss: {loss:.4f}")
    
    def predict(self, X):
        output = self.forward(X)
        if self.has_hidden:
            return np.round(output)
        else:
            return output
