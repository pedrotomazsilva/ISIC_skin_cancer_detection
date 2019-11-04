import torch
from model_1 import dice_coef_loss
import skimage.io as io
from image_utilities import limit, adjust_mask
import numpy as np
from sklearn.metrics import jaccard_similarity_score
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary
import sys
import time
import copy


'''
def loss_so_far(loss_batch, loss_sum, num_batches):
    loss_sum += loss_batch
    avg = loss_sum/num_batches
    sys.stdout.write("\rCurrent batch loss:%f" % loss_batch)
    sys.stdout.flush()
    return loss_sum, avg


def test_model(model, data, device, results_path, save, mask, loss_type, batch_size):

    model.eval()  # para avaliar o modelo - as layers de batch_norm e dropout ficam configuradas para validação e teste
    loss_sum = 0
    hits_sum = 0
    num_cancer = 0
    num_healthy = 0

    for num_batches, batch in enumerate(data):

        if 'dice' in loss_type:
            y_pred_batch = model((batch['image'].float()).to(device)).detach()
            dice_loss = dice_coef_loss(y_pred_batch, (batch['mask'].float()).to(device))
            loss_sum, dice_avg_loss = loss_so_far(dice_loss, loss_sum, num_batches+1)
        if 'jaccard' in loss_type:
            y_pred_batch = model((batch['image'].float()).to(device)).detach()
            jaccard_loss = jaccard(y_pred_batch, batch['mask'].float())
            loss_sum, jaccard_avg_loss = loss_so_far(jaccard_loss, loss_sum, num_batches+1)
        if 'mse_autoencoder' in loss_type:
            y_pred_batch = model((batch['image'].float()).to(device)).detach()
            mse = nn.MSELoss()
            mse_loss = mse(y_pred_batch, batch['image'].float().to(device))
            loss_sum, mse_avg = loss_so_far(mse_loss, loss_sum, num_batches+1)
        if loss_type == 'cross_entropy':
            y_pred_batch = model((batch['image'].float()).to(device)).detach()
            cross_entropy = nn.BCELoss()
            desease = batch['desease']
            desease[desease < 0] = 0  # Só se vai classificar melanoma vs non melanoma
            loss_sum += cross_entropy(y_pred_batch.view([-1]), desease.to(device).float())
            num_hits = hit(y_pred_batch.view([-1]), desease)
            hits_sum += num_hits
        if loss_type == 'compound':
            y_pred_batch, cancer = model((batch['image'].float()).to(device))
            loss_mask = dice_coef_loss(y_pred_batch.detach(), batch['mask'].float().to(device))
            #cross_entropy = nn.BCELoss()
            desease = batch['desease']
            desease[desease < 0] = 0  # Só se vai classificar melanoma vs non melanoma
            #loss_classification = cross_entropy(y_pred_batch.view([-1]), desease.to(device).float())
            num_cancer += desease
            num_healthy += (1 - desease)
            num_hits = hit(cancer.detach(), desease)
            print('\nnew_batch\nDesease:%d\nPredicted:%d' %(desease, cancer))
            hits_sum += num_hits
            loss = loss_mask
            loss_sum, dice_avg_loss = loss_so_far(loss, loss_sum, num_batches + 1)

        if save:
            save_results(y_pred_batch.detach(), batch, results_path, mask)

    if 'dice' in loss_type:
        return dice_avg_loss
    if 'mse_autoencoder' in loss_type:
        return mse_avg
    if 'jaccard' in loss_type:
        return jaccard_avg_loss

    if 'cross_entropy' in loss_type:
        print('Validation accuracy =', hits_sum/(batch_size*(num_batches+1)))
        return loss_sum/(num_batches+1)

    if 'compound' in loss_type:
        print('Validation accuracy =', hits_sum / (batch_size * (num_batches + 1)))
        print('num_cancer_cases:', num_cancer)
        print('num_healthy_cases:', num_healthy)
        return dice_avg_loss





def train_model(epochs_number, batch_size, model, loss_type, dataloader_train, dataloader_validation, optimizer, device,
                weights_name, save_path_weights='C:/Users/pedro/PycharmProjects/ISIC_skin_cancer/model_weights/'):

    #summary(model, input_size=(3, 256, 256))

    loss_train = []
    loss_validation = []
    num_batches = len(dataloader_train)

    for epoch in range(epochs_number):
        print('Epoch %d\n' % epoch)

        loss_sum = 0
        hits_sum = 0
        num_cancer = 0
        num_healthy = 0
        model.train()

        for num_samples, batch in enumerate(dataloader_train):

            input_batch = (batch['image'].float()).to(device)


            if loss_type == 'mse_autoencoder':
                y_pred = model(input_batch)
                mse = nn.MSELoss()
                loss = mse(y_pred, input_batch)
            elif loss_type == 'dice':
                y_pred = model(input_batch)
                loss = dice_coef_loss(y_pred, batch['mask'].float().to(device))
            elif loss_type == 'cross_entropy':
                y_pred = model(input_batch)
                cross_entropy = nn.BCELoss()
                desease = batch['desease']
                desease[desease < 0] = 0  # Só se vai classificar melanoma vs non melanoma
                loss = cross_entropy(y_pred.view([-1]), desease.to(device).float())
                num_hits = hit(y_pred, desease)
                hits_sum += num_hits
            elif loss_type == 'compound':
                y_pred, cancer = model(input_batch)
                loss_mask = dice_coef_loss(y_pred, batch['mask'].float().to(device))
                cross_entropy = nn.BCELoss()
                desease = batch['desease']
                desease[desease < 0] = 0  # Só se vai classificar melanoma vs non melanoma
                num_cancer += desease
                num_healthy += (1-desease)
                print('\n\ncacn', cancer.item())
                loss_classification = cross_entropy(cancer, desease.to(device).float())
                num_hits = hit(cancer, desease)
                hits_sum += num_hits
                loss_mask = 1+loss_mask  # 1-dice_coef
                normalized_loss_class = loss_classification/10
                loss = loss_mask + normalized_loss_class
                print('Dice_loss:', 1-loss_mask.item())
                print('loss_classification', normalized_loss_class.item())

            loss_sum, avg_loss_train = loss_so_far(loss, loss_sum, num_samples + 1)

            if (num_samples + 1) % 100 == 0:
                print('\n%d/%d batches ' % (num_samples+1, num_batches))
                print('Average loss so far:', avg_loss_train.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('\nTraining accuracy classification = ', hits_sum/(num_batches*batch_size))
        print('num_cancer_cases:', num_cancer)
        print('num_healthy_cases:', num_healthy)
        torch.save(model.state_dict(), save_path_weights + weights_name + '.pt')
        val_loss = test_model(model, dataloader_validation, device, save=False,
                              results_path=None, loss_type=loss_type, mask=True, batch_size=batch_size)

        print('Training loss:', avg_loss_train.item())
        print('Validation Loss:', val_loss)
        loss_train.append(avg_loss_train)
        loss_validation.append(val_loss)

    plot_results(epochs_number, loss_train, loss_validation)

    return loss_train, loss_validation





def hit(y_pred, desease):
    num_hits = 0
    for i, y in enumerate(y_pred):
        pred_i = round(y.item())

        if pred_i == desease[i].item():
            num_hits += 1

    return num_hits

'''


def save_results(y_pred_batch, batch, results_path, mask):

    y_pred_batch_npy = y_pred_batch.cpu().numpy()

    if mask:
        for mask in y_pred_batch_npy:

            mask = adjust_mask(mask)
            mask_path = results_path + batch['image_name'][0]
            io.imsave(mask_path, mask)
            img = batch['image'][0]
            img_contour = limit(mask, np.transpose(img.numpy(), (1, 2, 0)))
            img_contour_path = results_path + batch['image_name'][0][0:-3] + '_contour.jpg'
            io.imsave(img_contour_path, img_contour)
    else:
        for img in y_pred_batch_npy:
            img = np.transpose(img, (1, 2, 0))
            img_path = results_path + batch['image_name'][0]
            io.imsave(img_path, img)


def jaccard(input_batch, target_batch):
    total_loss = 0
    input_batch = input_batch.cpu()
    for i, input_img in enumerate(input_batch):
        target_img = target_batch.float()[i].view(-1).numpy()
        input_img = input_img.view(-1).numpy()
        input_img[input_img > np.mean(input_img)] = 1
        input_img[input_img <= np.mean(input_img)] = 0
        total_loss += jaccard_similarity_score(target_img.astype(np.int32), input_img.astype(np.int32))

    return total_loss / len(input_batch)


def train_model(model, dataloaders, optimizer, device,
                num_epochs, weights_name, scheduler=None):

    since = time.time()

    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                # scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for num_samples, batch in enumerate(dataloaders[phase]):

                num_batches = len(dataloaders[phase])
                input_imgs_batch = batch['image'].float().to(device)
                input_masks_batch = batch['mask'].float().to(device)
                labels_desease = batch['desease'].to(device).float()
                labels_desease[labels_desease < 0] = 0  # Só se vai classificar melanoma vs non melanoma
                batch_size = input_imgs_batch.shape[0]

                # zero the parameter gradients
                optimizer.zero_grad()

                pred_masks, pred_desease_pbs = model(input_imgs_batch)
                preds = torch.round(pred_desease_pbs)

                dice_loss = 1 + dice_coef_loss(pred_masks, input_masks_batch)  # 1-dice_coef
                binary_entropy = nn.BCELoss()
                norm_factor = 10
                binary_entropy_loss = binary_entropy(pred_desease_pbs[-1], labels_desease)/norm_factor
                loss = dice_loss + binary_entropy_loss

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * input_imgs_batch.size(0)
                running_corrects += torch.sum(preds == labels_desease)
                sys.stdout.write('\rRunning_Dice: %f Pred:%f Real:%f' %
                                 ((1-dice_loss).item(), pred_desease_pbs.item(),
                                  labels_desease.item()))
                sys.stdout.flush()

            epoch_loss = running_loss / (num_batches * batch_size)
            epoch_acc = float(running_corrects) / (num_batches * batch_size)
            torch.save(model.state_dict(), 'model_weights/' + weights_name + '.pt')

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                  phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                torch.save(model.state_dict(), 'model_weights/' + weights_name + '.pt')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
          time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model 


def plot_results(epochs_number, loss_train, loss_validation):
    x_axis = np.arange(epochs_number)
    plt.plot(x_axis, np.array(loss_train), 'r')
    plt.plot(x_axis, np.array(loss_validation), 'b')
    plt.show()
