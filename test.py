import torch
import torch.optim as optim
import torchvision.transforms as transforms
from utils import load_checkpoint
from get_loader import get_loader
from model import EncoderToDecoder
from PIL import Image

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
load_model_epoch = 50
TEST_IMG = '2903617548_d3e38d7f88.jpg'

encoder_input_sizes = {'InceptionV3': 299, 'AlexNet': 227, 'VGG': 244}  # TODO: Check if right for AlexNet and VGG

# TODO: CHOICES
ENCODER_CHOICE = 'InceptionV3'  # ['InceptionV3', 'AlexNet', 'VGG']

transform = transforms.Compose(
        [
            transforms.Resize((encoder_input_sizes[ENCODER_CHOICE], encoder_input_sizes[ENCODER_CHOICE])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Taken from ImageNet
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

model = EncoderToDecoder(ENCODER_CHOICE, EMBED_SIZE, HIDDEN_SIZE, VOCAB_SIZE, NUM_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if load_model:
    step = load_checkpoint(torch.load(f"{MODEL_TO_LOAD}my_checkpoint.pth.tar-epoch={load_model_epoch}"), model, optimizer)

# Try a test image -----------------------------------------------------------------------------------------------------
test_img = transform(Image.open(f"flickr8k/images/{TEST_IMG}").convert("RGB")).unsqueeze(0)
print("Example OUTPUT: " + " ".join(model.caption_image(test_img.to(device), train_dataset.vocab)))
