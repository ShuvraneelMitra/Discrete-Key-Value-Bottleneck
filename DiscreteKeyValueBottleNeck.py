import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize

class DiscreteKeyValueBottleneck(nn.Module):

    def __init__(self,
                 encoder=None,
                 num_codebooks=1,
                 enc_out_dim=None,
                 embed_dim=None,
                 value_dim=None,
                 keys_per_codebook=None,
                 requires_random_projection=True,
                 device=None):

        super().__init__()
        self.requires_random_projection = requires_random_projection
        self.encoder = encoder
        self.enc_out_dim = enc_out_dim if enc_out_dim is not None else embed_dim
        self.num_codebooks = num_codebooks

        self.values = nn.Parameter(torch.randn((num_codebooks, keys_per_codebook, value_dim),
                                               requires_grad=True,
                                               dtype=torch.float)).to(device)
        nn.init.xavier_normal_(self.values)

        self.rand_proj = torch.randn((num_codebooks, self.enc_out_dim, embed_dim)).to(device)

        self.quantizer = VectorQuantize(dim=embed_dim,
                                        codebook_size=keys_per_codebook,
                                        heads=num_codebooks,
                                        separate_codebook_per_head=True)

    def forward(self, batch):
        dec_out = []

        for x in batch:

            representations = self.encoder(x.unsqueeze(0)).squeeze(0) if self.encoder is not None else x

            if self.requires_random_projection:
                representations = representations.squeeze().unsqueeze(0)
                to_quantize = torch.matmul(representations, self.rand_proj)
                to_quantize = to_quantize.squeeze(0)
            else:
                to_quantize = representations

            to_quantize = to_quantize.squeeze().unsqueeze(0)

            quantized_outputs, indices, _ = self.quantizer(to_quantize)
            indices = torch.diagonal(indices.squeeze(0))

            mapped_values = self.values[torch.arange(self.num_codebooks), indices]

            average_pool = torch.mean(mapped_values, dim=1)
            dec_out.append(average_pool)

        return torch.stack(dec_out)
