# Some part of the code was referenced from below.
# https://github.com/pytorch/examples/tree/master/word_language_model 
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus
import os


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device set")

# Hyper-parameters
embed_size = 128
hidden_size = 2048
num_layers = 5
num_epochs = 10
num_samples = 500
batch_size = 20
seq_length = 50
learning_rate = 0.002

corpus = Corpus()
# load train data
data_dir = 'C:\\Users\\a.frost\\Desktop\\py\\Pytorch training\\pytorch-tutorial\\tutorials\\02-intermediate\\language_model\\data'
train_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.txt')]
ids_list = [corpus.get_data(file, batch_size) for file in train_files]
ids = torch.cat(ids_list, dim=1)

vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length

print("setting up model")
# model
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h):
        # Embed word ids to vectors
        x = self.embed(x)
        
        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)
        
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        
        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)

# Save both model and vocab info
def save_checkpoint(model, vocab_size, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size
    }, path)

# Load checkpoint and return vocab size
def load_checkpoint(path):
    checkpoint = torch.load(path)
    # Handle old format (direct state dict)
    if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
        return checkpoint, len(checkpoint['embed.weight'])
    # Handle new format
    return checkpoint['model_state_dict'], checkpoint['vocab_size']

# init and train
model_path = 'C:\\Users\\aaron\\Desktop\\VSC\\py\\pytorch-tutorial\\model.ckpt'
if os.path.exists(model_path):
    print('Loading existing model from', model_path)
    try:
        state_dict, saved_vocab_size = load_checkpoint(model_path)
        if saved_vocab_size != vocab_size:
            print(f'Warning: Current vocab size ({vocab_size}) differs from saved model ({saved_vocab_size})')
            vocab_size = saved_vocab_size
        model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)
        model.load_state_dict(state_dict)
        print('Model loaded successfully')
    except Exception as e:
        print(f'Error loading model: {e}')
        print('Training new model...')
        model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)
else:
    print('No existing model found. Training new model...')
    model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)
    # train
    for epoch in range(num_epochs):
        # Set initial hidden and cell states
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))
        
        for i in range(0, ids.size(1) - seq_length, seq_length):
            # Get mini-batch inputs and targets
            inputs = ids[:, i:i+seq_length].to(device)
            targets = ids[:, (i+1):(i+1)+seq_length].to(device)

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            def detach(states):
                return [state.detach() for state in states] 
            
            # Forward pass
            states = detach(states)
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape(-1))
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            step = (i+1) // seq_length
            if step % 100 == 0:
                print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                    .format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))

# Loss and opt 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states]

# Test
with torch.no_grad():
    with open('sample.txt', 'w', encoding="utf-8") as f:
        # Set intial hidden ane cell states
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))

        # Select one word id randomly
        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

        for i in range(num_samples):
            # Forward
            output, state = model(input, state)

            # Sample a word id
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()

            # Fill input with sampled word id
            input.fill_(word_id)

            # File write
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i+1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i+1, num_samples, 'sample.txt'))
                
# save_checkpoint(model, vocab_size, model_path)