import torch
import torch.nn as nn
import numpy as np
import time
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus
import os


# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device set")
embed_size = 128
hidden_size = 1024
num_layers = 5
num_epochs = 5
num_samples = 200
batch_size = 20
seq_length = 50
learning_rate = 0.002
model_path = 'C:\\Users\\a.frost\\Desktop\\py\\Pytorch training\\pytorch-tutorial\\model.ckpt'



corpus = Corpus()
data_dir = 'C:\\Users\\a.frost\\Desktop\\py\\Pytorch training\\pytorch-tutorial\\tutorials\\02-intermediate\\language_model\\data'
train_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.txt')]
ids_list = [corpus.get_data(file, batch_size) for file in train_files]
ids = torch.cat(ids_list, dim=1)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length



print("setting up model")
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        out = self.linear(out)
        return out, (h, c)

def save_checkpoint(model, vocab_size, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size
    }, path)

def load_checkpoint(path):
    checkpoint = torch.load(path)
    if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint:
        return checkpoint, len(checkpoint['embed.weight'])
    return checkpoint['model_state_dict'], checkpoint['vocab_size']



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
    #training
    start_time = time.time()
    model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)
    print('No existing model found. Training new model...')
    for epoch in range(num_epochs):
        states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
                  torch.zeros(num_layers, batch_size, hidden_size).to(device))
        
        for i in range(0, ids.size(1) - seq_length, seq_length):
            inputs = ids[:, i:i+seq_length].to(device)
            targets = ids[:, (i+1):(i+1)+seq_length].to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            def detach(states):
                return [state.detach() for state in states] 
            
            states = detach(states)
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            step = (i+1) // seq_length
            if step % 100 == 0:
                elapsed_time = time.time() - start_time
                print ( f'Training time: {elapsed_time/60:.2f} minutes / {elapsed_time:.2f} seconds', '     Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'.format(epoch+1, num_epochs, step, num_batches, loss.item(), np.exp(loss.item())))



criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
def detach(states):
    return [state.detach() for state in states]

# Testing
with torch.no_grad():
    with open('sample.txt', 'w', encoding="utf-8") as f:
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))

        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

        for i in range(num_samples):
            output, state = model(input, state)
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()
            input.fill_(word_id)

            # Write to file
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i+1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i+1, num_samples, 'sample.txt'))
                
save_checkpoint(model, vocab_size, model_path)