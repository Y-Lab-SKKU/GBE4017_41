# main.py
#%%
import torch
import yaml
from models.students_rnn import RNNNet
from trainer import CompetencyTaskTrainer
from competency_task_students import CompetencyTask

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# Load configuration
config = load_config()

# Configure CUDA if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Setup environment
env = CompetencyTask(
    dt=config['task']['dt'],
    timing=config['task']['timing']
)

# Initialize model
model = RNNNet(
    input_size=env.observation_space.shape[0],
    hidden_size=config['model']['hidden_size'],
    output_size=env.action_space.n,
).to(device)

# Train model
trainer = CompetencyTaskTrainer(model, env, config)
model = trainer.train()


# %%
