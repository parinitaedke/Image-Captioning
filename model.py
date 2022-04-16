import torch
import torch.nn as nn
import statistics
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, encoder_choice='InceptionV3', train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.encoder_choice = encoder_choice

        if encoder_choice == 'InceptionV3':
            self.model = models.inception_v3(pretrained=True, aux_logits=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, embed_size)

        elif encoder_choice == 'AlexNet':
            self.model = models.alexnet(pretrained=True)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, embed_size)

        elif encoder_choice == 'VGG':
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, embed_size)

        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # features = self.inception(images)
        features = self.model(images)
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, encoder_choice, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size, encoder_choice=encoder_choice)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)  # Word with highest probability
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)  # Embed the predicted word and use it for the next word

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]
