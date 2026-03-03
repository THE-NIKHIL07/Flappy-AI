import torch
from experiments.agent import DuelingDQN
import os

n_actions = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DuelingDQN(n_actions).to(device)

checkpoint = torch.load("models\checkpoint_ep200.pth", map_location=device)
model.load_state_dict(checkpoint)

model.eval()


dummy_input = torch.randn(1, 7).to(device)

os.makedirs("model", exist_ok=True)

torch.onnx.export(
    model,
    dummy_input,
    "model/dueling_dqn_model.onnx",
    input_names=["state"],
    output_names=["q_values"],
    dynamic_axes={"state": {0: "batch"}, "q_values": {0: "batch"}},
    opset_version=11
)

print("ONNX export successful!")