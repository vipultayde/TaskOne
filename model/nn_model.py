import torch
import torch.nn as nn
import numpy as np

class MedicalPredictor(nn.Module):
    """
    Simple Neural Network for Medical Prediction (e.g., Diabetes Risk)
    Input features: [glucose, blood_pressure, insulin, bmi, age, pregnancies, skin_thickness, diabetes_pedigree]
    Output: Probability of having diabetes (0-1)
    """
    def __init__(self, input_size=8, hidden_size=16, output_size=1):
        super(MedicalPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = nn.Linear(hidden_size//2, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification
        return x

    def get_parameters_as_int(self):
        params = {}
        for name, param in self.named_parameters():
            # Convert to int256 by scaling and rounding, flatten to 1D list
            scaled = (param.data * 1e6).round().int()
            params[name] = scaled.flatten().tolist()
        return params

    def set_parameters_from_int(self, params_dict):
        for name, param in self.named_parameters():
            if name in params_dict:
                int_values = params_dict[name]
                float_values = [x / 1e6 for x in int_values]
                param.data = torch.tensor(float_values, dtype=torch.float32).view(param.shape)

    @staticmethod
    def get_feature_names():
        return ['glucose', 'blood_pressure', 'insulin', 'bmi', 'age', 'pregnancies', 'skin_thickness', 'diabetes_pedigree']

    @staticmethod
    def preprocess_input(data_dict):
        """
        Preprocess input data dictionary to tensor
        Expected keys: glucose, blood_pressure, insulin, bmi, age, pregnancies, skin_thickness, diabetes_pedigree
        """
        features = MedicalPredictor.get_feature_names()
        input_data = []
        for feature in features:
            if feature in data_dict:
                input_data.append(float(data_dict[feature]))
            else:
                input_data.append(0.0)  # Default value if missing

        return torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
