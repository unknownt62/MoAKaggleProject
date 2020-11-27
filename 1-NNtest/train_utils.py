"""Training utilities."""

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class Flatten(nn.Module):
    """A custom layer that views an input as 1D."""

    def forward(self, input):
        return input.view(input.size(0), -1)


def batchify_data(x_data, y_data, batch_size):
    """Takes a set of data points and labels and groups them into batches."""
    # Only take batch_size chunks (i.e. drop the remainder)
    N = int(len(x_data) / batch_size) * batch_size
    batches = []
    for i in range(0, N, batch_size):
        batches.append({
            'x': torch.tensor(x_data[i:i + batch_size],
                              dtype=torch.float32),
            'y': torch.tensor(y_data[i:i + batch_size],
                               dtype=torch.float32)
        })
    return batches


def compute_accuracy(predictions, y):
    """Computes the accuracy of predictions against the gold labels, y."""
    return np.mean(np.equal(predictions.numpy(), y.numpy()))


def train_model(train_data, dev_data, model, lr=0.01, momentum=0.9, nesterov=False, n_epochs=30):
    """Train a model for N epochs given data and hyper-params."""
    # We optimize with SGD
    # (TODO) "Often, just replacing vanilla SGD with an optimizer like Adam or RMSProp will boost performance noticably.""

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)

    for epoch in range(1, n_epochs + 1):
        print("-------------\nEpoch {}:\n".format(epoch))

        # Run **training***
        loss, acc = run_epoch(train_data, model.train(), optimizer)
        print('Train | loss1: {:.6f}  accuracy1: {:.6f} | loss2: {:.6f}  accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

        # Run **validation**
        val_loss, val_acc = run_epoch(dev_data, model.eval(), optimizer)
        print('Valid | loss1: {:.6f}  accuracy1: {:.6f} | loss2: {:.6f}  accuracy2: {:.6f}'.format(val_loss[0], val_acc[0], val_loss[1], val_acc[1]))

        # Save model
        torch.save(model, 'MoA_fully_connected.pt')


def run_epoch(data, model, optimizer):
    """Train model for one pass of train data, and return loss, acccuracy"""
    # Gather losses
    losses_first_label = []
    losses_second_label = []
    batch_accuracies_first = []
    batch_accuracies_second = []

    # If model is in train mode, use optimizer.
    is_training = model.training

    # Iterate through batches
    for batch in tqdm(data):
        # Grab x and y
        x, y= batch['x'], batch['y']

        # Get output predictions for both the upper and lower numbers
        out1= model(x) #.items()

        # Predict and store accuracy
        predictions_first_label = torch.argmax(out1, dim=205)

        batch_accuracies_first.append(compute_accuracy(predictions_first_label, y[0]))


        # Compute both losses
        loss1 = F.cross_entropy(out1, y[0])
   
        losses_first_label.append(loss1.data.item())
  

        # If training, do an update.
        if is_training:
            optimizer.zero_grad()
            joint_loss = 0.5 * (loss1)
            joint_loss.backward()
            optimizer.step()

    # Calculate epoch level scores
    avg_loss = np.mean(losses_first_label), np.mean(losses_second_label)
    avg_accuracy = np.mean(batch_accuracies_first), np.mean(batch_accuracies_second)
    return avg_loss, avg_accuracy
