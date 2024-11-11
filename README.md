# RNN Implementation Assignment

## Assignment Overview
This assignment tasks you with implementing a Recurrent Neural Network (RNN) to solve a competency evaluation task. You will implement a model that learns to compare the "competency" of two eels based on their interactions with fish in their environment.

### Learning Objectives
- Understand and implement core RNN architectures
- Analyze model learning and performance
- Gain hands-on experience with PyTorch

### Your Tasks

1. **RNN Implementation **
   - Complete the `RNN` class in `models/students_rnn.py`
   - Required implementations:
     - `__init__`: Initialize the network layers
     - `init_hidden`: Create the initial hidden state
     - `forward`: Implement the forward pass
   - Complete the `RNNNet` class that uses your RNN
     - Combine the RNN with appropriate output layers

2. **Testing and Debugging**
   - Verify your implementation works with the provided training loop
   - Debug any issues in your RNN implementation
   - Test whether your model actually learns the task


### Tips for Success
1. **Start with the Basics**
   - Make sure you understand the RNN forward pass
   - Test each component separately
   - Use print statements or debugger to verify tensor shapes

2. **Debugging Steps**
   - Check tensor dimensions at each step
   - Verify hidden state updates
   - Monitor loss during training

3. **Common Pitfalls**
   - Incorrect tensor dimensions
   - Missing nonlinearities
   - Hidden state initialization issues
   - Gradient problems (exploding/vanishing)

4. **Going Beyond**
   - Try different architectures
   - Experiment with hyperparameters
   - Analyze model behavior with different input patterns


### Testing Your Implementation


### **Debug Common Issues**
- Loss not decreasing: Check gradients and learning rate
- Exploding gradients: Verify gradient clipping
- No learning: Check model predictions vs random

### Performance Analysis

Document your findings in a report including:

1. Learning Curves
   - Training loss
   - Validation accuracy
   - Learning speed

2. Model Behavior
   - Does it learn both phases of the task?
   - How does it handle different eel behaviors?
   - What patterns does it recognize?

3. Failure Analysis (if applicable)
   - Where does the model struggle?
   - What debugging steps did you try?
   - Potential improvements?

---


# Competency Task Neural Network Project

This project implements a competency task environment using neural networks to train an agent to evaluate and compare "eel competency" based on fish behavior patterns.

## Project Structure
```
.
├── config.yaml              # Configuration settings
├── environment.yml          # Conda environment specifications
├── main.py                 # Main training script
├── models/                 # Models directory **ASSIGNMENT**
│   └── students_rnn.py  # RNN model implementation
├── trainer.py              # Training loop implementation
└── competency_task.py      # Environment implementation
```

## Installation

### Prerequisites
- Anaconda or Miniconda installed on your system
  - If you don't have it installed, download from: https://docs.conda.io/en/latest/miniconda.html

### Setting up the environment

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate the conda environment from the environment.yml file:
```bash
# Create new environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate competency_task  # or whatever name is specified in environment.yml
```

3. Install neurogym (required for cognitive tasks):
```bash
# Clone neurogym repository
git clone https://github.com/neurogym/neurogym.git

# Install neurogym in editable mode
cd neurogym
pip install -e .
cd ..
```

4. Verify the installation:
```bash
# Check if key packages are installed
python -c "import torch; print(torch.__version__)"
python -c "import gymnasium; print(gymnasium.__version__)"
python -c "import neurogym; print(neurogym.__version__)"
```

## Running the Project

1. Make sure your environment is activated:
```bash
conda activate competency_task
```

2. Train the model:
```bash
python main.py
```

## Project Configuration

The project uses `config.yaml` for configuration settings. You can modify:
- Model architecture parameters
- Training hyperparameters
- Task environment settings
- Optimization settings

Feel free to add more configuration parameters if you feel like your model needs it. 


