import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN

import numpy as np
import pandas as pd
from pathlib import Path

# Clear GPU cache
torch.cuda.empty_cache()

encoder_input_sizes = {'InceptionV3': 299, 'AlexNet': 227, 'VGG': 244}  # TODO: Check if right for AlexNet and VGG

# TODO: CHOICES
ENCODER_CHOICE = 'AlexNet'  # ['InceptionV3', 'AlexNet', 'VGG']

load_model = False
save_model = True
train_CNN = False  # If False, only fine-tune. If True, train the full CNN

# TODO: Update hyperparameters as need be
EMBED_SIZE = 512  # [256, 512]
HIDDEN_SIZE = 256  # [256, 512]
NUM_LAYERS = 1  #
LEARNING_RATE = 3e-4
NUM_EPOCHS = 50  # [25, 50]
BATCH_SIZE = 256  # [128, 256]
NUM_WORKERS = 2


# Determines the OUTPUT FOLDER
OUTPUT_FOLDER = f"OUTPUTS/encoder={ENCODER_CHOICE}-embed_size={EMBED_SIZE}-hidden_size={HIDDEN_SIZE}-num_layers={NUM_LAYERS}-lr={LEARNING_RATE}-epochs={NUM_EPOCHS}-bs={BATCH_SIZE}-num_workers={NUM_WORKERS}-trainCNN={train_CNN}/"
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((encoder_input_sizes[ENCODER_CHOICE], encoder_input_sizes[ENCODER_CHOICE])),
            # transforms.RandomCrop(encoder_input_sizes[ENCODER_CHOICE]),  # Depending on which encoder we use, we crop images accordingly.
            # # This is data augmentation of sorts since we are doing random crop
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    # Get the loaders and datasets for train-val-test
    train_loader, train_dataset = get_loader(
        root_folder="flickr8k/images",
        annotation_file="flickr8k/train_captions.txt",
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    val_loader, val_dataset = get_loader(
        root_folder="flickr8k/images",
        annotation_file="flickr8k/val_captions.txt",
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    test_loader, test_dataset = get_loader(
        root_folder="flickr8k/images",
        annotation_file="flickr8k/test_captions.txt",
        transform=transform,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Get vocab size
    VOCAB_SIZE = len(train_dataset.vocab)

    # for tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(ENCODER_CHOICE, EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE, NUM_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Only finetune the CNN
    # for name, param in model.encoderCNN.inception.named_parameters():
    for name, param in model.encoderCNN.model.named_parameters():

        # Determines name of layer we want to tune in respective encoder architectures
        if ENCODER_CHOICE == 'InceptionV3':
            weight = "fc.weight"
            bias = "fc.bias"
        elif ENCODER_CHOICE == "AlexNet" or ENCODER_CHOICE == "VGG":
            weight = "classifier.6.weight"
            bias = "classifer.6.bias"

        if weight in name or bias in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    # TODO: CHECK WHERE YOU ARE LOADING MODELS FROM
    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    # Store the epoch loss for training and validation sets
    train_mean_loss, val_mean_loss = [], []

    for epoch in range(NUM_EPOCHS):
        # Uncomment the line below to see a couple of test cases
        print_examples(model, device, train_dataset)

        # Set the model to train mode
        model.train()
        train_batch_loss = []

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint, output_folder=OUTPUT_FOLDER, filename=f"my_checkpoint.pth.tar-epoch={epoch}")

        print(f"Training epoch: {epoch}")
        for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
            imgs = imgs.to(device)
            captions = captions.to(device)

            # print(captions)

            outputs = model(imgs, captions[:-1])  # Want the model to predict the END token so we don't send the last one in
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            # print(outputs)
            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            # Back prop
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            train_batch_loss.append(loss.item())

            # Clear GPU cache
            torch.cuda.empty_cache()

        train_mean_loss.append(np.mean(train_batch_loss))

        # Validation
        val_batch_loss = []

        # Set the model to evaluation mode
        model.eval()
        print(f"Validation epoch: {epoch}")
        for idx, (imgs, captions) in tqdm(enumerate(val_loader), total=len(val_loader), leave=False):
            imgs = imgs.to(device)
            captions = captions.to(device)

            with torch.no_grad():
                outputs = model(imgs, captions[:-1])  # Want the model to predict the END token so we don't send the last one in
                loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar("Validation loss", loss.item(), global_step=step)

            val_batch_loss.append(loss.item())

        val_mean_loss.append(np.mean(val_batch_loss))

        # # del for GPU cleaning?
        # del(train_loader)
        # del(val_loader)

        # Save the training and validation loss
        loss_df = pd.DataFrame({"train_mean_loss": train_mean_loss, "val_mean_loss": val_mean_loss})
        loss_csvfile = f"{OUTPUT_FOLDER}Loss.csv"
        loss_df.to_csv(loss_csvfile)


if __name__ == "__main__":
    train()
