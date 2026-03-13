import torch

def kl_div_normal(output, target_mean=0.0, target_std=1.0):
    """
    Compute the KL divergence between the output distribution and a target normal distribution.
    """
    
    # Check input dimensions
    if output.dim() != 2:
        raise ValueError("Output tensor must be of shape [batch_size, n]")

    # Compute mean and standard deviation of the output vectors
    output_mean = output.mean(dim=1)
    output_std = output.std(dim=1)

    # Prevent standard deviation from being zero to avoid numerical instability
    output_std = torch.clamp(output_std, min=1e-8)

    # Compute KL divergence
    kl = 0.5 * ((output_std / target_std).pow(2) + ((output_mean - target_mean) / target_std).pow(2) - 1 - torch.log((output_std / target_std).pow(2)))
    
    # Compute mean KL divergence over the batch
    kl_mean = kl.mean()

    return kl_mean


def kl_div_normal_muvar(mu, logvar):
    """
    Compute the KL divergence between the output distribution and a standard normal distribution using mean and log-variance.
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kl


def masked_rec_loss(pred, gt, mask, reduction='mean'):
    loss = torch.norm((pred - gt)[mask], dim=-1)
    if reduction == 'mean':
        return loss.sum() / (mask.sum() + 1e-6) # in m
    elif reduction == 'sum':
        return loss.sum()
    else:
        loss = torch.sum(torch.norm((pred - gt) * mask.unsqueeze(-1), dim=-1), dim=-1) / (mask.sum(dim=-1) + 1e-6)
        return loss