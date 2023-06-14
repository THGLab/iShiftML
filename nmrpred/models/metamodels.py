"""
These are the meta-models that combine encoders and decoders in different ways 
"""

import torch
from torch import nn

class Sequential(nn.Module):
    def __init__(self, modules):
        '''
        A basic sequential model that connects each component in a sequential manner
        '''
        super().__init__()
        self.components = nn.ModuleList(modules)

    def forward(self, inputs):
        x = inputs
        for i in range(len(self.components)):
            x = self.components[i](x)
        return x


class Attention(nn.Module):
    def __init__(self, structure_encoder, attention_mask_network, mol_based=False, with_low_level_input=False, return_attention_masks=False):
        """
        An attention based network that uses the structure-encoded latent vector to apply attention masks to the tensorial inputs
        """
        super().__init__()
        self.structure_encoder = structure_encoder
        self.attention_mask_network = attention_mask_network
        self.mol_based = mol_based
        self.with_low_level_input = with_low_level_input
        self.return_attention_masks = return_attention_masks



    def forward(self, inputs):
        tensorial_inputs = inputs["tensor_features"]
        if hasattr(self, "with_low_level_input"):
            if self.with_low_level_input:
                low_level_inputs = inputs["low_level_inputs"]
                tensorial_inputs = torch.cat([tensorial_inputs, low_level_inputs[:, None]], dim=-1)
        encoded_structures = self.structure_encoder(inputs)
        if self.mol_based:
            masks = inputs["M"]
            tensorial_inputs = tensorial_inputs[masks]
            encoded_structures = encoded_structures[masks]
        attention_inputs = torch.cat([encoded_structures, tensorial_inputs], dim=-1)
        attention_masks = self.attention_mask_network(attention_inputs)
        final_predictions = torch.sum(tensorial_inputs * attention_masks[:, :-1], dim=-1) + attention_masks[:, -1]
        return final_predictions

	# if self.return_attention_masks:
        #    return final_predictions, attention_masks
        # else:
        #    return final_predictions

class FeatureSupplementary(nn.Module):
    def __init__(self, structure_encoder, output_network):
        """
        A network that combines the latent vector coming from the structure encoder and the tensorial features and connect with a MLP output network
        """
        super().__init__()
        self.structure_encoder = structure_encoder
        self.output_network = output_network

    def forward(self, inputs):
        tensorial_inputs = inputs["tensorial_features"]
        encoded_structures = self.structure_encoder(inputs)
        concatenated_hidden = torch.cat([encoded_structures, tensorial_inputs], dim=-1)
        final_predictions = self.output_network(concatenated_hidden)
        return final_predictions


class ExtendedAttention(nn.Module):
    def __init__(self, structure_encoder, attention_mask_network, value_network):
        """
        An attention based network that uses the structure-encoded latent vector to apply attention masks to the 
        value_network transformed tensorial inputs
        """
        super().__init__()
        self.structure_encoder = structure_encoder
        self.attention_mask_network = attention_mask_network
        self.value_network = value_network



    def forward(self, inputs):
        tensorial_inputs = inputs["tensorial_features"]
        encoded_structures = self.structure_encoder(inputs)
        # if self.mol_based:
        #     masks = inputs["M"]
        #     tensorial_inputs = tensorial_inputs[masks]
        #     encoded_structures = encoded_structures[masks]
        attention_inputs = torch.cat([encoded_structures, tensorial_inputs], dim=-1)
        attention_masks = self.attention_mask_network(attention_inputs)
        values = self.value_network(tensorial_inputs)
        final_predictions = torch.sum(values * attention_masks[:, :-1], dim=-1) + attention_masks[:, -1]
        return final_predictions
