
import torch

model_start = torch.load("../int_base/pytorch_model.bin")

model_end = torch.load("Test_runs/ft_testing_freeze/checkpoint-1600/pytorch_model.bin")

print(model_end["bert.encoder.layer.0.intermediate.dense.weight"])

print(model_start["classifier.weight"])