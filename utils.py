import numpy as np

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))

    for i, val in enumerate(y):
        ary[i, val] = 1
    
    return ary

# Generates minibatches
def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])

    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]

# MSE loss function
def mse_loss(targets, probas, num_labels=10):
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)

    print(f'target shape: {targets}')
    print(f'probas shape: {probas}')
    return np.mean((onehot_targets - probas) ** 2)

def accuracy(targets, predicted_labels):
    return np.mean(targets == predicted_labels)

def get_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0

    minibatch_gen = minibatch_generator(X, y, minibatch_size)

    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)

        predicted_labels = np.argmax(probas, axis=1)

        onehot_targets = int_to_onehot(targets, num_labels)

        loss = np.mean((onehot_targets - probas) ** 2)

        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss

    mse = mse / i
    acc = correct_pred / num_examples

    return mse, acc
