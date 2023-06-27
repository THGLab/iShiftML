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





class Attention_TEV(nn.Module):
    def __init__(self, structure_encoder, attention_mask_network, dim_tev_off = 2, mol_based=False, with_low_level_input=False, return_attention_masks=False):
        """
        A Network like attention but with a TEV input replacing the tensorial features
        """
        super().__init__()
        self.structure_encoder = structure_encoder
        self.attention_mask_network = attention_mask_network
        self.mol_based = mol_based
        self.with_low_level_input = with_low_level_input
        self.return_attention_masks = return_attention_masks
        self.dim_tev_off = dim_tev_off

    def forward(self, inputs):
        tev_inputs = inputs["tev"]
        if hasattr(self, "with_low_level_input"):
            if self.with_low_level_input:
                low_level_inputs = inputs["low_level_inputs"]
                tev_inputs = torch.cat([tev_inputs, low_level_inputs[:, None]], dim=-1)
        encoded_structures = self.structure_encoder(inputs)
        if self.mol_based:
            masks = inputs["M"]
            tev_inputs = tev_inputs[masks]
            encoded_structures = encoded_structures[masks]
        attention_inputs = torch.cat([encoded_structures, tev_inputs[:,self.dim_tev_off:]], dim=-1)
        attention_masks = self.attention_mask_network(attention_inputs)
        final_predictions = tev_inputs[:,0] * attention_masks[:, 0] + tev_inputs[:, 1] * attention_masks[:, 1]  + attention_masks[:, 2]

        if self.return_attention_masks:
            return final_predictions, attention_masks
        else:
            return final_predictions



class Attention_TEV_SVD(nn.Module):
    def __init__(self, structure_encoder, DIA_attention, PARA_attention, Bias_attention, dim_1tensor = 80,mol_based=False, with_low_level_input=False, return_attention_masks=False):
        """
        A Network like attention but with a TEV input replacing the tensorial features
        """
        super().__init__()
        self.structure_encoder = structure_encoder
        self.DIA_attention = DIA_attention
        self.PARA_attention = PARA_attention
        self.Bias_attention = Bias_attention
        self.mol_based = mol_based
        self.with_low_level_input = with_low_level_input
        self.return_attention_masks = return_attention_masks
        self.dim_1tensor = dim_1tensor

    def forward(self, inputs):
        tev_inputs = inputs["tev"]
        encoded_structures = self.structure_encoder(inputs)
        if self.mol_based:
            masks = inputs["M"]
            tev_inputs = tev_inputs[masks]
            encoded_structures = encoded_structures[masks]
        
        # Here we do the attention for DIA1, DIA2, DIA3, PARA1, PARA2, PARA3, and Bias
        # First we extract the  DIA1, DIA2, DIA3, PARA1, PARA2, PARA3, and overall information from the TEV inputs
        DIA1 = tev_inputs[:, 6: 6+ self.dim_1tensor]
        DIA2 = tev_inputs[:, 6+ self.dim_1tensor: 6+ 2 * self.dim_1tensor]
        DIA3 = tev_inputs[:, 6+ 2 * self.dim_1tensor: 6+ 3 * self.dim_1tensor]
        PARA1 = tev_inputs[:, 6+ 3 * self.dim_1tensor: 6+ 4 * self.dim_1tensor]
        PARA2 = tev_inputs[:, 6+ 4 * self.dim_1tensor: 6+ 5 * self.dim_1tensor]
        PARA3 = tev_inputs[:, 6+ 5 * self.dim_1tensor: 6+ 6 * self.dim_1tensor]
        Overall_info = tev_inputs[:, 6 + 6 * self.dim_1tensor: ]
        DIA = DIA1+DIA2+DIA3
        PARA = PARA1+PARA2+PARA3

        # Then we do the attention for DIA1, DIA2, DIA3, PARA1, PARA2, PARA3, and Bias
        DIA1_attention_masks = self.DIA_attention(torch.cat([encoded_structures, DIA1, DIA2+DIA3, PARA, Overall_info], dim=-1))
        DIA2_attention_masks = self.DIA_attention(torch.cat([encoded_structures, DIA2, DIA1+DIA3, PARA, Overall_info], dim=-1))
        DIA3_attention_masks = self.DIA_attention(torch.cat([encoded_structures, DIA3, DIA1+DIA2, PARA, Overall_info], dim=-1))
        PARA1_attention_masks = self.PARA_attention(torch.cat([encoded_structures, PARA1, PARA2+PARA3, DIA, Overall_info], dim=-1))
        PARA2_attention_masks = self.PARA_attention(torch.cat([encoded_structures, PARA2, PARA1+PARA3, DIA, Overall_info], dim=-1))
        PARA3_attention_masks = self.PARA_attention(torch.cat([encoded_structures, PARA3, PARA1+PARA2, DIA, Overall_info], dim=-1))
        Bias_attention_masks = self.Bias_attention(torch.cat([encoded_structures, DIA, PARA, Overall_info], dim=-1))
        #Concatenate the attention masks
        attention_masks = torch.cat([DIA1_attention_masks, DIA2_attention_masks, DIA3_attention_masks, PARA1_attention_masks, PARA2_attention_masks, PARA3_attention_masks, Bias_attention_masks], dim=-1)

        # Finally we combine the attention masks and tev_inputs[:, :6] to get the final prediction
        final_predictions = torch.sum(tev_inputs[:, :6] * attention_masks[:, :-1], dim=-1) + attention_masks[:, -1]

        if self.return_attention_masks:
            return final_predictions, attention_masks
        else:
            return final_predictions
