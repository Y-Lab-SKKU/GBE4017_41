import torch
import torch.nn as nn

class RNN(nn.Module):
    """A basic RNN for the competency task.
    
    This RNN should:
    1. Process the input observations (fixation, eel positions, fish positions)
    2. Maintain a hidden state that integrates information over time
    3. Help make decisions about which eel is more competent
    """
    def __init__(self, input_size, hidden_size):
        """Initialize the RNN.
        
        Args:
            input_size (int): Total dimension of input vector
            hidden_size (int): Number of hidden units in RNN
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # TODO: Create the core RNN components
        

    def init_hidden(self, batch_size, device):
        """Initialize the hidden state with zeros.
        
        Args:
            batch_size (int): Size of batch
            device: Device to create tensor on (cpu/cuda)
            
        Returns:
            torch.Tensor: Initial hidden state
        """
        # TODO: Students: Initialize the hidden state
        # Should return zeros with shape (batch_size, hidden_size)
        pass

    def forward(self, input, hidden=None):
        """Process input and update hidden state.
        
        Args:
            input (torch.Tensor): Input sequence [sequence_length, batch_size, input_size]
            hidden (torch.Tensor, optional): Initial hidden state
            
        Returns:
            tuple: (outputs, final_hidden_state)
            - outputs: sequence of hidden states for each input
            - final_hidden_state: last hidden state
        """
        # TODO: Students: Implement the forward pass
        # 1. Initialize hidden state if None provided
        # 2. Process the sequence one timestep at a time:
        # 3. Collect outputs for each timestep
        pass


class RNNNet(nn.Module):
    """Full network combining RNN and output layer for decision making."""
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the full network.
        
        Args:
            input_size (int): Input dimension
            hidden_size (int): RNN hidden state dimension
            output_size (int): Number of output classes (actions)
        """
        super().__init__()
        # TODO: Students: Create the RNN and output layers


    def forward(self, x):
        """Process input sequence and generate action predictions.
        
        Args:
            x (torch.Tensor): Input sequence [sequence_length, batch_size, input_size]
            
        Returns:
            tuple: (outputs, rnn_states)
            - outputs: action predictions for each timestep
            - rnn_states: RNN hidden states for analysis
        """
        # TODO: Students: Implement forward pass

        pass