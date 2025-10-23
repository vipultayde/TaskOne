import torch
import torch.nn as nn
import numpy as np

class HeartDiseasePredictor(nn.Module):
    def __init__(self, input_size=13, hidden_size=8, output_size=1):
        super(HeartDiseasePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def get_parameters_as_int(self):
        params = {}
        for name, param in self.named_parameters():
            # Convert to int256 by scaling and rounding
            scaled = (param.data * 1e6).round().int()
            params[name] = scaled.tolist()
        return params

    def set_parameters_from_int(self, params_dict):
        for name, param in self.named_parameters():
            if name in params_dict:
                int_values = params_dict[name]
                float_values = [x / 1e6 for x in int_values]
                param.data = torch.tensor(float_values, dtype=torch.float32).view(param.shape)
