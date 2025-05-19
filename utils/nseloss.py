import torch


class NSELoss(torch.nn.Module):
    """Calculate (batch-wise) NSE Loss.

    Each sample i is weighted by 1 / (std_i + eps)^2, where std_i is the standard deviation of the 
    discharge from the basin, to which the sample belongs.

    Parameters:
    -----------
    eps : float
        Constant, added to the weight for numerical stability and smoothing, default to 0.1
    """

    def __init__(self, eps: float = 0.00001):
        super(NSELoss, self).__init__()
        eps = torch.tensor(eps, dtype=torch.float32)
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor):
        """Calculate the batch-wise NSE Loss function.

        Parameters
        ----------
        y_pred : torch.Tensor
            Tensor containing the network prediction.
        y_true : torch.Tensor
            Tensor containing the true discharge values
        q_stds : torch.Tensor
            Tensor containing the discharge std (calculate over training period) of each sample

        Returns
        -------
        torch.Tenor
            The (batch-wise) NSE Loss
        """
        squared_error = (y_pred - y_true) ** 2
        self.eps = self.eps.to(q_stds.device)
        weights = 1 / (q_stds + self.eps) ** 2
        weights = weights.reshape(-1, 1, 1)
        scaled_loss = weights * squared_error
        return scaled_loss
        # Ngage = y_true.shape[0]
        # losssum = 0
        # nsample = 0
        # for ii in range(Ngage):
        #     p0 = y_pred[ii, :, 0]
        #     t0 = y_true[ii, :, 0]
        #     mask = t0 == t0
        #     if len(mask[mask == True]) > 0:
        #         p = p0[mask]
        #         t = t0[mask]
        #         tmean = t.mean()
        #         SST = torch.sum((t - tmean) ** 2) + self.eps
        #         SSRes = torch.sum((t - p) ** 2)
        #         temp = 1 - SSRes / SST
        #         losssum = losssum + temp
        #         nsample = nsample + 1
        # # minimize the opposite average NSE
        # loss = -(losssum / nsample)
        # return loss
# class NSELoss(torch.nn.Module):
#     def __init__(self):
#         super(NSELoss, self).__init__()
#
#     def forward(self, output, target):
#         Ngage = target.shape[1]
#         losssum = 0
#         nsample = 0
#         for ii in range(Ngage):
#             p0 = output[:, ii, 0]
#             t0 = target[:, ii, 0]
#             mask = t0 == t0
#             if len(mask[mask == True]) > 0:
#                 p = p0[mask]
#                 t = t0[mask]
#                 tmean = t.mean()
#                 SST = torch.sum((t - tmean) ** 2)
#                 if SST != 0:
#                     SSRes = torch.sum((t - p) ** 2)
#                     temp = 1 - SSRes / SST
#                     losssum = losssum + temp
#                     nsample = nsample + 1
#         # minimize the opposite average NSE
#         loss = -(losssum / nsample)
#         return loss