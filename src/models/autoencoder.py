import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, n_layers: int = 1, dropout: float = 0.0):
        super(LSTMAutoencoder, self).__init__()
        
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # -------------------
        # ENCODER
        # -------------------
        # Takes the sequence (batch, seq_len, features) and processes it through time
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )

        # -------------------
        # DECODER
        # -------------------
        # Takes the compressed representation and expands it back
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim, 
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Maps the decoder's hidden state back to the original 130 physical signals
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # 1. Encode the input sequence
        # We only care about the final hidden state `h_n`, which represents the entire sequence
        _, (h_n, _) = self.encoder(x)
        
        # Get the hidden state from the last layer of the LSTM
        last_hidden = h_n[-1]  # Shape: (batch_size, hidden_dim)
        
        # 2. Prepare Decoder Input
        # We repeat that single hidden state for every time step in the sequence
        seq_len = x.shape[1]
        decoder_input = last_hidden.unsqueeze(1).repeat(1, seq_len, 1) # Shape: (batch, seq_len, hidden_dim)
        
        # 3. Decode
        decoder_out, _ = self.decoder(decoder_input)
        
        # 4. Reconstruct to original features
        reconstruction = self.output_layer(decoder_out) # Shape: (batch, seq_len, n_features)
        
        return reconstruction

if __name__ == "__main__":
    # Quick sanity check to ensure the tensor math works before we train
    logging_setup = True
    print("Testing LSTM Autoencoder architecture...")
    
    dummy_batch_size = 32
    dummy_seq_len = 250
    dummy_features = 130 
    
    model = LSTMAutoencoder(n_features=dummy_features, hidden_dim=64, n_layers=2)
    dummy_input = torch.randn(dummy_batch_size, dummy_seq_len, dummy_features)
    
    output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    if dummy_input.shape == output.shape:
        print("✅ Architecture is structurally sound. Input shape matches output shape.")
    else:
        print("❌ Error: Output shape mismatch.")