import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

num_classes = 26
batch_size = 1
# batch_size = 5
# batchify(raw_data, batch_size)
num_classes = 26      # the number of possible classes we have (the labels tensors is between 0 and 4)
input_size = 26       # one-hot encoded vector dimensions
hidden_size = 26      # we use 5 dimensional hidden state vectors to directly predict the character
embedding_size = 10

class Network(nn.Module):
    def __init__(self, hidden_size, embedding_size):
        super(Network, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        self.rnn = nn.RNN(input_size=embedding_size,
                          hidden_size=self.hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        #h_0 = Variable(torch.zeros(1, embedding_size, self.hidden_size))
        h_0 = Variable(torch.zeros(1, 1, self.hidden_size))
        emb = self.embedding(x)
        emb = emb.view(batch_size, embedding_size, -1)
        # Propagate embedding through RNN
        # Input: (batch, seq_len, embedding_size)
        # h_0: (num_layers * num_directions, batch, hidden_size)
        # breakpoint()
        out, _ = self.rnn(emb.view(1,x.size()[1],-1), h_0)
        return self.fc(out)


model = Network(hidden_size, embedding_size)
print(model)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


# inputs = torch.LongTensor(ast.literal_eval(data[0]))[:-1].unsqueeze(0)
# target = torch.LongTensor(ast.literal_eval(data[0]))[1:]
inputs = [[ 1,  2,  8, 13, 14, 17,  3, 20, 24, 23, 11, 22, 11, 25, 11, 23, 24, 20,
          3, 21,  3, 18,  3, 17, 14, 13, 14, 17,  3, 20, 24, 23, 11, 25, 11, 22,
         19,  3,  2,  7,  2,  4,  2,  8, 13,  8,  2,  5,  2,  9, 15,  9]]
labels = [ 2,  8, 13, 14, 17,  3, 20, 24, 23, 11, 22, 11, 25, 11, 23, 24, 20,  3,
        21,  3, 18,  3, 17, 14, 13, 14, 17,  3, 20, 24, 23, 11, 25, 11, 22, 19,
         3,  2,  7,  2,  4,  2,  8, 13,  8,  2,  5,  2,  9, 15,  9, 16]
inputs = torch.LongTensor(inputs)
labels = torch.LongTensor(labels)
inputs = Variable(inputs)
labels = Variable(labels)

for epoch in range(1000):
    outputs = model(inputs)
    breakpoint()
    outputs = outputs.view(-1,num_classes)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    print(idx)
    # result_str = [idx2char[c] for c in idx.squeeze()]
    # if (epoch%20 == 0) or (epoch == 99):
    #   print("epoch: %d, loss: %1.3f" % (epoch + 1, loss.data[0]))
    #   print("Predicted string: ", ''.join(result_str))

print("Learning finished!")