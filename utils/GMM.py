import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
import math

class GMM2(nn.Module):
    def __init__(
            self,
            args, 
            num_prototypes : int, 
            stage_num  : int = 5, 
            momentum : float = 0.9, 
            beta : float = 1.0,
            lambd : float = 0.1, 
            eps : float = 1e-6,
    ):
        super(GMM2, self).__init__()
        self.num_prototypes = num_prototypes
        self.input_dim = args.input_dim
        self.stage_num = stage_num
        self.momentum = momentum 
        self.beta = beta
        self.lambd = lambd
        self.eps = eps

        prototypes = torch.Tensor(num_prototypes, self.input_dim)
        nn.init.kaiming_uniform_(prototypes, a = math.sqrt(5))
        prototypes = prototypes / (1e-6 + prototypes.norm(dim=1, keepdim=True))
        
        self.prototypes = nn.Parameter(prototypes.clone())
        self.register_buffer("running_prototypes", prototypes.clone())
        
    def _normalize_prototypes(self):
        with torch.no_grad():
            norm = self.prototypes.norm(dim=1, keepdim=True) + self.eps
            self.prototypes.data = self.prototypes.data / norm
            
    def forward(self, cls : torch.Tensor, is_train = True):
        assert cls.dim() == 2
        b, d = cls.shape
        device = cls.device
        
        self._normalize_prototypes()
        local_proto = self.prototypes.unsqueeze(0).expand(b, -1, -1).contiguous()
        _cls = cls 
        
        for _ in range(self.stage_num):
            latent = torch.einsum("bd, bkd->bk", cls, local_proto)
            latent = latent / self.lambd
            r = F.softmax(latent, dim = 1)
            
            new_proto = torch.mm(r.t(), _cls)
            new_proto = new_proto / (new_proto.norm(dim=1, keepdim=True) + self.eps)
            local_proto = new_proto.unsqueeze(0).expand(b, -1, -1).contiguous()

        dot = torch.einsum("bd,bkd->bk", cls, local_proto)
        dot = dot / self.lambd
        r = F.softmax(dot, dim=1)

        z_recon = torch.mm(r, new_proto)
        z_out = self.beta * z_recon + cls

        recon_loss = F.mse_loss(z_recon, cls)
        entropy_loss = - (r * torch.log(r + self.eps)).sum(dim = 1).mean()
        proto_similarity = torch.mm(new_proto, new_proto.t())
        eye = torch.eye(self.num_prototypes, device = device)
        diversity_loss = torch.mean(torch.abs(proto_similarity * (1 - eye)))

        if is_train:
            with torch.no_grad():
                old_proto = self.running_prototypes.to(device)
                updated = self.momentum * old_proto + (1 - self.momentum) * new_proto 
                updated = updated / (updated.norm(dim = 1, keepdim = True) + self.eps)
                self.running_prototypes.copy_(updated.detach())
                
                update_ratio = 0.01 
                self.prototypes.data = (1 - update_ratio) * self.prototypes.data + update_ratio * new_proto.detach()

        return r, z_out, {
            'recon_loss': recon_loss,
            'entropy_loss': entropy_loss,
            'diversity_loss': diversity_loss
        }