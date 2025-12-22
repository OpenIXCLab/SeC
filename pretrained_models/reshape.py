import torch
import torch.nn.functional as F

length = 22
model = f"pretrained_models/sam2.1_hiera_large.pt"
A = torch.load(model, map_location='cpu', weights_only=True)

embedding = A["model"]["maskmem_tpos_enc"] # [num_maskmem, 1, 1, self.mem_dim]
embedding = embedding.permute(1, 3, 2, 0)  # [1, self.mem_dim, 1, num_maskmem]

upsampled = F.interpolate(embedding, size=(1, length), mode='bilinear', align_corners=True)
upsampled = upsampled.permute(3, 2, 0, 1)  # [num_maskmem, 1, 1, self.mem_dim]

A["model"]["maskmem_tpos_enc"] = upsampled

torch.save(A, f"pretrained_models/sam2.1_hiera_large_maskmem_{length}.pt")