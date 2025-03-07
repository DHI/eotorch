import torch
import torch.nn as nn


class GaussianNLL(nn.Module):
    """
    Gaussian negative log likelihood to fit the mean and variance to p(y|x)
    Note: We estimate the heteroscedastic variance. Hence, we include the var_i of sample i in the sum
    over all samples N. Furthermore, the constant log term is discarded.
    """
    def __init__(self, reduction : str = 'mean'):
        super().__init__()
        self.eps = 1e-8
        self.reduction = reduction

    def __call__(self, mean : torch.Tensor, variance : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
        """
        The exponential activation is applied already within the network to directly output variances.

        Parameters:
            mean (torch.Tensor): 
                Predicted mean values.
            variance (torch.Tensor): 
                Predicted variance.
            target (torch.Tensor): 
                Ground truth labels.

        Returns:
            torch.Tensor: 
                Gaussian negative log likelihood
        """       
        variance = variance + self.eps
        if self.reduction == 'mean':
            return torch.mean(0.5 / variance * (mean - target)**2 + 0.5 * torch.log(variance))
        elif self.reduction == 'none':
            return 0.5 / variance * (mean - target)**2 + 0.5 * torch.log(variance)