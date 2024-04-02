from utils import minibatch_generator, get_mse_and_acc
import matplotlib.pyplot as plt

def train(model, X_train, y_train, X_val, y_val, num_epochs, learning_rate=0.1):
    epoch_loss = []
    epoch_train_acc = []
    epoch_val_acc = []

    # Iterate epochs
    for e in range(num_epochs):
        minibatch_gen = minibatch_generator(
            X_train, 
            y_train, 
            minibatch_size=100
        )

        # Iterate training examples
        for X_train_mini, y_train_mini in minibatch_gen:
            a_h, a_out = model.forward(X_train_mini)

            # Compute gradients
            d_loss__d_w_out, d_loss__d_b_out, \
            d_loss__d_w_h, d_loss__d_b_h = \
            model.backward(
                X_train_mini, 
                a_h, 
                a_out, 
                y_train_mini
            )

            # Update weights
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
            model.weight_out -= learning_rate * d_loss__d_w_out
            model.bias_out -= learning_rate * d_loss__d_b_out

        # Log epoch mse and acc
        train_mse, train_acc = get_mse_and_acc(model, X_train, y_train)
        _, val_acc = get_mse_and_acc(model, X_val, y_val)

        train_acc, val_acc = train_acc*100, val_acc*100

        epoch_train_acc.append(train_acc)
        epoch_val_acc.append(val_acc)
        epoch_loss.append(train_mse)

        print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
                f'| Train MSE: {train_mse:.2f} '
                f'| Train Acc: {train_acc:.2f}% '
                f'| Val acc: {val_acc:.2f}%')
    
    plt.plot(range(len(epoch_loss)), epoch_loss)
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()

    plt.plot(range(len(epoch_train_acc)), epoch_train_acc, label='training')
    plt.plot(range(len(epoch_val_acc)), epoch_val_acc, label='validation')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()
    
    return epoch_loss, epoch_train_acc, epoch_val_acc
