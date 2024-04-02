import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils import sigmoid, int_to_onehot, get_mse_and_acc
from train import train

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X = X.values
y = y.astype(int).values

# Normalize pixel values
X = (( X / 255. ) - .5) * 2

# Split temp train and test data
X_temp, X_test, y_temp, y_test = train_test_split(
    X, 
    y, 
    test_size=10000, 
    random_state=1, 
    stratify=y
)

# Further split train data into validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, 
    y_temp, 
    test_size=5000, 
    random_state=1, 
    stratify=y_temp
)

class Neural_net:
    def __init__(self, num_features, num_hidden, num_classes, random_seed=1):
        super().__init__()
        self.num_classes = num_classes

        rng = np.random.RandomState(random_seed)

        self.weight_h = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(num_hidden, num_features)
        )

        self.bias_h = np.zeros(num_hidden)

        self.weight_out = rng.normal(
            loc=0.0,
            scale=0.1,
            size=(num_classes, num_hidden)
        )

        self.bias_out = np.zeros(num_classes)
    
    def forward(self, x):
        z_h = np.dot(x, self.weight_h.T) + self.bias_h

        a_h = sigmoid(z_h)

        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out

        a_out = sigmoid(z_out)

        return a_h, a_out
    
    def backward(self, x, a_h, a_out, y):
        y_onehot = int_to_onehot(y, self.num_classes)

        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]

        d_a_out__d_z_out = a_out * (1. - a_out)

        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        d_z_out__dw_out = a_h

        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        d_z_out__a_h = self.weight_out

        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        d_a_h__d_z_h = a_h * (1. - a_h)

        d_z_h__d_w_h = x

        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)

        d_loss_d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss_d_b_h)

model = Neural_net(num_features=28*28, num_hidden=50, num_classes=10)

np.random.seed(123)

epoch_loss, epoch_train_acc, epoch_val_acc = train(
    model, 
    X_train, 
    y_train, 
    X_val, 
    y_val, 
    num_epochs=50, 
    learning_rate=0.1
)

X_test_subset = X_test[:1000, :]
y_test_subset = y_test[:1000]

_, probas = model.forward(X_test_subset)

test_pred = np.argmax(probas, axis=1)

misclassified_imgs = X_test_subset[y_test_subset != test_pred][:25]
misclassified_labels = test_pred[y_test_subset != test_pred][:25]
correct_labels = y_test_subset[y_test_subset != test_pred][:25]

fig, ax = plt.subplots(
    nrows=5, 
    ncols=5, 
    sharex=True, 
    sharey=True, 
    figsize=(8,8)
)

ax = ax.flatten()

for i in range(25):
    img = misclassified_imgs[i].reshape(28,28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title(f'{i+1} '
      f'True: {correct_labels[i]}\n'
      f'Predicted: {misclassified_labels[i]}'                
    )

ax[0].set_xticks([])
ax[0].set_yticks([])

plt.tight_layout()
plt.show()