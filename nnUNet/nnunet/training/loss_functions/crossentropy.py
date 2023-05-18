from torch import nn, Tensor
import torch


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]

############################################################# Only for BHSD
############################################################# another shape issue
        
        # Specify the desired shape of the output tensor
        desired_shape = tuple(input.shape)[:1] + tuple(input.shape)[2:]

        # Calculate the number of zeros to add in the 3rd dimension
        num_zeros = desired_shape[1] - target.size(1)

        # Generate the tensor with zeros to add
        zeros_tensor = torch.zeros((desired_shape[0], num_zeros, desired_shape[2], desired_shape[3]), device=target.device)

        # Concatenate the input tensor and zeros tensor along the 3rd dimension
        target = torch.cat((target, zeros_tensor), dim=1)

############################################################## Only for BHSD

        return super().forward(input, target.long())

