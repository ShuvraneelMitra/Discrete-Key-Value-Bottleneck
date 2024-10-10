import torch
import torch.nn as nn

def qutil(vector, matrix):
    
    distances = torch.norm(matrix - vector, dim=1)
    min_idx = torch.argmin(distances)
    min_row = matrix[min_idx]
    
    return min_row, min_idx

class DiscreteKeyValueBottleneck(nn.Module):
    
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
        self.rand_proj = torch.randn((self.enc_out_dim,
                                      num_codebooks, 
                                      embed_dim))

    def initialize_random_keys(self, num_codebooks, keys_per_codebook, embed_dim, lower=0, upper=1):
        
        z =  torch.rand((num_codebooks, keys_per_codebook, embed_dim))
        return lower + (upper - lower) * z
    
    def quantize(self, to_quantize, keys):
        
        quantized_outputs = []
        indices = []
        for i in range(self.num_codebooks):
            row, index = qutil(to_quantize[i], keys[i])
            quantized_outputs.append(row)
            indices.append(index)
        
        return torch.stack(quantized_outputs), torch.tensor(indices)
        
    def forward(self, x):
        dec_out = []
        
        for input in x:  
            representations = self.encoder(input) if self.encoder is not None else input
            
            if self.requires_random_projection:
                to_quantize = representations * self.rand_proj
            else:
                to_quantize = representations
            
            quantized_outputs, indices = self.quantize(to_quantize, self.keys)
            
            mapped_values = []
            for i, idx in enumerate(indices):
                mapped_values.append(self.values[i][idx])
            mapped_values = torch.stack(mapped_values)
            
            average_pool = torch.mean(mapped_values, dim=0)
            decoded_values = nn.functional.softmax(average_pool, dim=0)
        
            dec_out.append(decoded_values)
            
        return torch.stack(dec_out)
            
            