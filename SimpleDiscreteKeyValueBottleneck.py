import torch
import torch.nn as nn

def qutil(vectors, matrix):
    if vectors.dim() <= 1:
        vectors = vectors.unsqueeze(0)
        
    distances = torch.cdist(vectors, matrix, p=2)
    
    assert distances.shape == (vectors.shape[0], matrix.shape[0]), f"L2 norm matrix mismatch, should have been {(vectors.shape[0], matrix.shape[0])} but got {distances.shape}"
    # So distances[i][j] = L2(vectors[i] : vector, matrix[j]  : vector). So min_indices should be of size vectors.shape[0] so 
    # take the argmin along dim=1, i.e. along the row for each row.
    
    min_indices = torch.argmin(distances, dim=1)
    best_rows = matrix[min_indices]
    
    return best_rows, min_indices


class SimpleDiscreteKeyValueBottleneck(nn.Module):
    
    def __init__(self,
                 encoder = None,
                 num_codebooks = 1,
                 enc_out_dim = None,
                 embed_dim = None,
                 value_dim = None,
                 keys_per_codebook = None,
                 naive = False,
                 requires_random_projection = False 
                 ):
        super().__init__()
        self.requires_random_projection = requires_random_projection
        self.encoder = encoder
        self.enc_out_dim = enc_out_dim if enc_out_dim is not None else embed_dim
        self.num_codebooks = num_codebooks
        self.keys = self.initialize_random_keys(num_codebooks,
                                                keys_per_codebook,
                                                embed_dim)
        self.values = nn.Parameter(torch.randn((num_codebooks,
                                               keys_per_codebook, 
                                               value_dim), requires_grad=True,
                                               dtype=float))
        nn.init.xavier_normal_(self.values)
        self.rand_proj = torch.randn((num_codebooks, 
                                      self.enc_out_dim,
                                      embed_dim))

    def initialize_random_keys(self, num_codebooks, keys_per_codebook, embed_dim, lower=0, upper=1):
        
        z =  torch.rand((num_codebooks, keys_per_codebook, embed_dim))
        return lower + (upper - lower) * z
    
    def quantize(self, to_quantize, keys):
        quantized_outputs = []
        indices = []
        if self.num_codebooks > 1:
            for i in range(self.num_codebooks):
                row, index = qutil(to_quantize[i], keys[i])
                quantized_outputs.append(row)
                indices.append(index)
        else:
            row, index = qutil(to_quantize, keys[0])
            quantized_outputs.append(row)
            indices.append(index)
        
        return torch.stack(quantized_outputs, dim=0), torch.tensor(indices)
        
    def forward(self, batch):
        dec_out = []
        
        for x in batch:  
            representations = self.encoder(x) if self.encoder is not None else x
            
            if self.requires_random_projection:
                representations = representations.view(1, representations.shape[0], 1)
                to_quantize = torch.matmul(self.rand_proj, representations)
                to_quantize = to_quantize.squeeze(2)
            else:
                to_quantize = representations
            
            quantized_outputs, indices = self.quantize(to_quantize, self.keys)
            
            mapped_values = []
            for i, idx in enumerate(indices):
                mapped_values.append(self.values[i][idx])
            mapped_values = torch.stack(mapped_values)
            
            average_pool = torch.mean(mapped_values, dim=0)
            
            dec_out.append(average_pool)
            
        return torch.stack(dec_out)
