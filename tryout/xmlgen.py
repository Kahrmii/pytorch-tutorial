import os
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import random

# Verzeichnis mit XML-Dateien
xml_dir = 'C:\\Users\\aaron\\Desktop\\VSC\\py\\pytorch-tutorial'

# XML-Dateien lesen und parsen
data = []
for filename in os.listdir(xml_dir):
    if filename.endswith('.xml'):
        try:
            tree = ET.parse(os.path.join(xml_dir, filename))
            root = tree.getroot()
            xml_str = ET.tostring(root, encoding='unicode')
            data.append(xml_str)
        except ET.ParseError:
            print(f"Error parsing {filename}, skipping this file.")

# Tokenisierung
all_tokens = ' '.join(data).split()
vocab = Counter(all_tokens)
vocab = {token: idx for idx, token in enumerate(vocab.keys(), start=1)}
vocab['<pad>'] = 0  # Padding Token

# Umwandeln der Daten in Token-Indices
data_indices = [[vocab[token] for token in xml.split()] for xml in data]
max_length = max(len(xml) for xml in data_indices)

# Padding
data_indices = [xml + [0] * (max_length - len(xml)) for xml in data_indices]
data_indices = torch.tensor(data_indices)

# Trainings- und Validierungsdaten
train_data = data_indices[:2]
val_data = data_indices[2:]

# Modellarchitektur
class XMLGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout=0.5):
        super(XMLGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        if x.size(0) > 0 and x.size(1) > 0:
            x = self.batch_norm(x.contiguous().view(-1, x.size(2)))
            x = self.dropout(x)
            x = self.fc(x)
            return x.view(x.size(0), -1, x.size(1)), hidden
        else:
            return x, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_().to(device),
                weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_().to(device))

# Hyperparameter
vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 256
num_layers = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modell, Loss-Funktion und Optimizer
model = XMLGenerator(vocab_size, embed_dim, hidden_dim, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
num_epochs = 100
batch_size = 1  # Ensure batch size is consistent
seq_length = 30  # Ensure sequence length is consistent

def train_model():
    for epoch in range(num_epochs):
        hidden = model.init_hidden(batch_size)
        for i in range(0, len(train_data) - seq_length, seq_length):
            inputs = train_data[:, i:i+seq_length].to(device)
            targets = train_data[:, i+1:i+seq_length+1].to(device)

            # Vorwärtspropagation
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

            # Rückpropagation und Optimierung
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Hidden State für den nächsten Schritt resetten
            hidden = (hidden[0].detach(), hidden[1].detach())

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generierung
def generate_xml(model, start_token, max_length=100):
    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden(1)
        input_tensor = torch.tensor([[start_token]]).to(device)
        generated = []

        for _ in range(max_length):
            output, hidden = model(input_tensor, hidden)
            output = output.squeeze().argmax().item()
            generated.append(output)
            input_tensor = torch.tensor([[output]]).to(device)  # Update input_tensor for next iteration

        return generated

# Example usage
start_token = vocab['<CATALOG>']  # Using <CATALOG> as the start token
generated_xml = generate_xml(model, start_token)
print(generated_xml)

# Postprocessing
def tokens_to_xml(tokens, vocab_inv):
    return ' '.join([vocab_inv[token] for token in tokens if token != 0])

# Vokabular (Token zu Wort-Abbildung)
vocab_inv = {idx: token for token, idx in vocab.items()}
generated_xml_str = tokens_to_xml(generated_xml, vocab_inv)
print(generated_xml_str)

# Evaluation
def evaluate_model():
    val_data = val_data.to(device)
    hidden = model.init_hidden(val_data.size(0))  # Verwende batch_size aus val_data

    with torch.no_grad():
        outputs, hidden = model(val_data, hidden)  # Pass the correctly shaped tensor
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), val_data.reshape(-1))
        print(f'Validation Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), 'model.ckpt')

# Funktion zur Eingabeaufforderung
def prompt_user():
    while True:
        print("\nOptions:")
        print("1. Train model")
        print("2. Generate XML")
        print("3. Evaluate model")
        print("4. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            train_model()
        elif choice == '2':
            start_token = vocab['<CATALOG>']  # Assuming <CATALOG> token is in the vocabulary
            generated_xml = generate_xml(model, start_token)
            generated_xml_str = tokens_to_xml(generated_xml, vocab_inv)
            print(generated_xml_str)
        elif choice == '3':
            evaluate_model()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")

# Hauptprogramm
if __name__ == "__main__":
    prompt_user()