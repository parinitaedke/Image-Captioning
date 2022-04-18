import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN
from PIL import Image

import numpy as np
import pandas as pd
from pathlib import Path

from model import CNNtoRNN

# TODO: CHOICES
ENCODER_CHOICE = 'InceptionV3'  # ['InceptionV3', 'AlexNet', 'VGG']

load_model = True
save_model = True
train_CNN = False  # If False, only fine-tune. If True, train the full CNN

# TODO: Update hyperparameters as need be
EMBED_SIZE = 512  # [256, 512]
HIDDEN_SIZE = 512  # [256, 512]
NUM_LAYERS = 1  #
LEARNING_RATE = 3e-4
NUM_EPOCHS = 150  # [25, 50]
BATCH_SIZE = 128  # [128, 256]
NUM_WORKERS = 2

MODEL_TO_LOAD = f"OUTPUTS/encoder={ENCODER_CHOICE}-embed_size={EMBED_SIZE}-hidden_size={HIDDEN_SIZE}-num_layers={NUM_LAYERS}-lr={LEARNING_RATE}-epochs={NUM_EPOCHS}-bs={BATCH_SIZE}-num_workers={NUM_WORKERS}-trainCNN={train_CNN}/"

encoder_input_sizes = {'InceptionV3': 299, 'AlexNet': 227, 'VGG': 244}  # TODO: Check if right for AlexNet and VGG

# TODO: CHOICES
ENCODER_CHOICE = 'InceptionV3'  # ['InceptionV3', 'AlexNet', 'VGG']

transform = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.RandomCrop(encoder_input_sizes[ENCODER_CHOICE]),
        # Depending on which encoder we use, we crop images accordingly.
        # This is data augmentation of sorts since we are doing random crop
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Get vocab size
VOCAB_SIZE = len(train_dataset.vocab)


model = CNNtoRNN(ENCODER_CHOICE, EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE, NUM_LAYERS).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if load_model:
    step = load_checkpoint(torch.load(f"{MODEL_TO_LOAD}my_checkpoint.pth.tar"), model, optimizer)

test_img = transform(Image.open("flickr8k/images/2903617548_d3e38d7f88.jpg").convert("RGB")).unsqueeze(0)
print("Example OUTPUT: " + " ".join(model.caption_image(test_img.to(device), train_dataset.vocab)))

