import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GraphEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(GraphDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x[:, :6]

class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GraphAutoencoder, self).__init__()
        self.encoder = GraphEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = GraphDecoder(latent_dim, hidden_dim, input_dim)

    def forward(self, data):
        z = self.encoder(data)
        
        # Select the corresponding columns from the decoder output
        decoder_output = self.decoder(z)
        output = decoder_output[:, :data.x.shape[1]]
        
        return output

