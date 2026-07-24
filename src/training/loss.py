import torch
from torch import nn
import torch.nn.functional as F

from src.configs import DatasetInfo
from src.datasets.encoding import Encoding

def reconstruction_loss_fn(
    *,
    x: tuple,
    x_predicted: tuple,
    encoding: Encoding,
    dataset_info: DatasetInfo,
    device: torch.device
) -> tuple:
    """
    Compute the reconstruction loss for the CVAE model.
    The reconstruction loss is computed as the sum of the losses for each component of the input:
    - Categorical attributes: Binary Cross Entropy Loss
    - Numerical attributes: Mean Squared Error Loss
    - Activities: Binary Cross Entropy Loss
    - Timestamps: Mean Squared Error Loss
    - Resources: Binary Cross Entropy Loss
    Args:
        x_predicted: The predicted input from the CVAE model.
        x: The original input.
        encoding: The vocabularies the batch is encoded with.
        dataset_info: The dataset information.
        device: The device to run the computations on.
    Returns:
        loss: The total reconstruction loss.
        loss_components: A tensor containing the individual losses for each component.
    """
    # Initialize the loss components
    cat_attributes_loss, num_attributes_loss, cf_loss, ts_loss, resource_loss = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)

    # Define the loss functions for each component
    cat_attribute_loss_fn = nn.BCELoss(reduction='sum')
    num_attribute_loss_fn = nn.MSELoss(reduction='sum')
    cf_loss_fn = nn.BCELoss(reduction='sum')
    ts_loss_fn = nn.MSELoss(reduction='sum')
    resource_loss_fn = nn.BCELoss(reduction='sum')

    # Unpack the predicted and original inputs
    attributes_predicted, activities_predicted, ts_predicted, resources_predicted = x_predicted
    attributes, activities, timestamps, resources = x

    # Convert categorical attributes to one-hot encoding
    for attribute_name, attribute_val in attributes.items():
        if attribute_name in encoding.s2i: # it is a categorical attribute
            attributes[attribute_name] = F.one_hot(attribute_val.to(torch.int64), num_classes=len(encoding.s2i[attribute_name])).to(torch.float32).to(device)

    # Turn all PAD activities into EOT activities, since PAD sits outside the output
    # width of the model and so cannot appear in the ground truth we compare against
    activities = torch.where(activities == encoding.pad_activity_index, encoding.eot_activity_index, activities).to(device)
    # Convert activities to one-hot encoding
    activities = F.one_hot(activities.to(torch.int64), num_classes=dataset_info.num_activities).to(torch.float32)

    # attributes
    for attribute_name, attribute_val in attributes.items():
        if attribute_name in encoding.s2i:
            cat_attributes_loss += cat_attribute_loss_fn(attributes_predicted[attribute_name], attribute_val)
        else:
            attributes_predicted[attribute_name] = attributes_predicted[attribute_name].squeeze()
            num_attributes_loss += num_attribute_loss_fn(attributes_predicted[attribute_name], attribute_val).to(torch.float32)

    # activities
    cf_loss += cf_loss_fn(activities_predicted, activities)

    # ts
    # the length of a trace is where its first EOT activity sits, read off the
    # ground truth so that the mask does not depend on how good the model currently is
    lens = activities.argmax(dim=2).argmax(dim=1)
    # use lens and compute loss only for the first lens elements
    for i in range(len(lens)):
        ts_predicted[i, lens[i]:] = 0.0
        
    ts_loss += ts_loss_fn(ts_predicted, timestamps)

    # resources
    resources = torch.where(resources == encoding.pad_resource_index, encoding.eot_resource_index, resources).to(device)
    resources = F.one_hot(resources.to(torch.int64), num_classes=dataset_info.num_resources).to(torch.float32)

    resource_loss += resource_loss_fn(resources_predicted, resources)

    # sum up loss components
    loss = cat_attributes_loss + num_attributes_loss + cf_loss + ts_loss + resource_loss

    return loss, torch.tensor([cat_attributes_loss, num_attributes_loss, cf_loss, ts_loss, resource_loss])

def kl_divergence_loss_fn(
    mean: torch.Tensor,
    std: torch.Tensor
) -> torch.Tensor:
    """
    Compute the KL divergence loss between two Gaussian distributions.
    """
    # Ensure that the std is positive to avoid numerical issues
    std = torch.clamp(std, min=1e-9)
    # Compute the KL divergence loss using the formula for two Gaussian distributions
    return - torch.sum(1 + torch.log(std ** 2) - mean ** 2 - std ** 2)