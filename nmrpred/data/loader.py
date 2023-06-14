import torch

def batch_dataset_converter(input, device):
    result = {}
    data_type_mapper = {"inputs": torch.float32,
                        "targets": torch.float32,
                        "low_level_inputs": torch.float32,
                        "R": torch.float32,
                        "Z": torch.long,
                        "N": torch.long,
                        "M": torch.bool,
                        "NM": torch.long,
                        "AM": torch.long,
                        "RM": torch.float32,
                        "aev": torch.float32,
                        "cs_scaler": torch.float32,
                        "D": torch.float32,
                        "V": torch.float32,
                        "tensor_features": torch.float32}
    for key in input:
        if key == "labels":
            result["labels"] = input["labels"]
            continue
        if key == "tensorial_features":
            result["tensor_features"] = torch.tensor(input["tensorial_features"],
                                                     dtype=torch.float32,
                                                     device=device)
            continue
        if key in data_type_mapper:
            result[key] = torch.tensor(input[key],
            dtype=data_type_mapper[key],
            device=device)
    
    return result
