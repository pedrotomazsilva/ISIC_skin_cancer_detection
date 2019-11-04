from model_hibrid import Unet
from data_processing import get_data_loaders
from model_autoencoder import Autoencoder, Decoder, Encoder
import torch
from training_utilities import train_model


batch_size = 1
learning_rate = 1e-4
epochs_number = 30

save_path = 'C:/Users/pedro/PycharmProjects/ISIC_skin_cancer/'
dataloader_train, dataloader_validation, dataloader_test = get_data_loaders(save_path, batch_size,
                                                                            mask_type='mask')
dataloaders = {
    'train': dataloader_train,
    'val': dataloader_validation,
    'test': dataloader_test
}

device = torch.device("cuda:0")
model = Unet(batch_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


model.load_state_dict(torch.load(save_path + '/model_weights/unet_weights_model_hibrid.pt'))

'''
pretrained_weights = torch.load(save_path + '/model_weights/model_autoencoder.pt')
encoder = Encoder().to(device)
encoder_state_dict = encoder.state_dict()
pretrained_weights_items = list(pretrained_weights.items())


# Transfer weights from de encoder
count = 0
for key, value in encoder_state_dict.items():
    layer_name, weights = pretrained_weights_items[count]
    encoder_state_dict[key] = weights
    count += 1


for num_layer, layer in enumerate(encoder.children()):
    if num_layer < 16:
        for params in layer.parameters():
            params.requires_grad = False
'''
train_model(num_epochs=epochs_number, model=model, dataloaders=dataloaders, optimizer=optimizer,
            device=device, weights_name='unet_weights_model_hibrid')






