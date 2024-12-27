import os
import torch
import optuna
import numpy as np

def train_one_epoch(model, loss_fn, optimizer, training_loader, device):
    """
    Single epoch training
    """
    # start training
    model.train(True)
    # print frequency
    print_freq = len(training_loader)//5
    # cummulative statistics
    running_cum_loss = 0.0
    running_cum_size = 0
    # iterate over batches
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        x_ppi, x_ph, x_o, labels = data
        x_ppi, x_ph, x_o, labels = x_ppi.to(device), x_ph.to(device), x_o.to(device), labels.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Make predictions for this batch
        outputs = model(x_ppi, x_ph, x_o)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels.long())
        loss.backward()
        optimizer.step()
        last_mean_loss = loss.item()
        # Aggregate cumulative
        running_cum_loss += last_mean_loss * labels.shape[0]
        running_cum_size += labels.shape[0]
        # print progress
        if i % print_freq == 0:
            print(f"\t\tbatch {i} of {len(training_loader)}, loss = {running_cum_loss/running_cum_size:.3f}")
    # Return of the average over the whole training set
    return running_cum_loss / running_cum_size


def evaluate_model(model, loss_fn, data_loader, device, return_predictions = False):
    """
    Evaluate current model on given data loader. Returns loss and accuracy
    """
    # do not train
    model.train(False)
    # cumulative values
    running_cum_loss = 0.0
    running_cum_correct = 0
    running_cum_size = 0
    running_predictions = []
    for i, vdata in enumerate(data_loader):
        vx_ppi, vx_ph, vx_o, vlabels = vdata
        vx_ppi, vx_ph, vx_o, vlabels = vx_ppi.to(device), vx_ph.to(device), vx_o.to(device), vlabels.to(device)
        with torch.no_grad():
            voutputs = model(vx_ppi, vx_ph, vx_o)
            vloss = loss_fn(voutputs, vlabels.long()).item()
            voutputs = voutputs.argmax(1)
        # Aggregate cumulative
        running_cum_loss += vloss * vlabels.shape[0]
        running_cum_correct += (voutputs == vlabels).float().sum()
        running_cum_size += vlabels.shape[0]
        if return_predictions:
            running_predictions.append(voutputs.cpu().numpy())
    # concat predictions
    if running_predictions:
        running_predictions = np.concatenate(running_predictions)
    # return
    return running_cum_loss / running_cum_size, running_cum_correct / running_cum_size, running_predictions


def train_and_evaluate(model, max_epochs, loss_fn, optimizer, training_loader, validation_loader, device, model_save_dir, early_stopping_epochs = 5, trial = None):
    """
    The overall training function
    """
    # prepare save name
    model_save_name = os.path.join(model_save_dir, f"optuna_temp_trial_{trial.number if trial else -1}_best_model.pt")
    # prepare tracking
    epochs = []
    train_loss = []
    validation_loss = []
    validation_acc = []
    best_vloss = 1000000.
    epochs_from_best = 0
    for epoch in range(max_epochs):
        print('\tEPOCH {}:'.format(epoch + 1))
        # Indicate training phase
        model.train(True)
        # One training epoch
        avg_loss = train_one_epoch(model, loss_fn, optimizer, training_loader, device)
        # Indicate evaluation phase
        model.train(False)
        # Validation performance
        vloss, vacc, _ = evaluate_model(model, loss_fn, validation_loader, device)
        # collect values
        epochs.append(epoch)
        train_loss.append(avg_loss)
        validation_loss.append(vloss)
        validation_acc.append(vacc)
        print(f"\tTRAIN loss: {avg_loss:.3f}, VALIDATION loss: {vloss:.3f}, accuracy: {vacc:.3f}")
        # Track best performance
        if vloss < best_vloss:
            best_vloss = vloss
            best_vacc = vacc
            epochs_from_best = 0  
            # saving the temporary best model
            torch.save(model.state_dict(), model_save_name)
        else:
            print("\tNo improvement in this epoch.")
            epochs_from_best += 1
        # early stopping and returning best accuraccy
        if epochs_from_best > early_stopping_epochs:
            print("\tEarly stopping now.")
            # loading best model
            model.load_state_dict(torch.load(model_save_name))
            break
        # Optuna add prune mechanism
        if trial:
            trial.report(best_vacc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return best_vloss, best_vacc

