import torch
import torch.nn as nn
import numpy as np
import neurogym as ngym
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

class CompetencyTaskTrainer:
    def __init__(
        self, 
        model, 
        env, 
        config=None
    ):
        """Initialize the trainer with basic parameters.
        
        Args:
            model: The neural network model
            env: The CompetencyTask environment
            config: Configuration dictionary. If None, uses default values
        """
        self.model = model
        self.env = env
        
        # Default configuration
        default_config = {
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'num_epochs': 5,
                'num_trials_per_epoch': 200,
                'val_frequency': 1,
                'validation_trials': 100,
                'gradient_clip': 1.0,
                'optimizer': {
                    'type': 'adamw',
                    'weight_decay': 1e-4
                }
            }
        }
        
        # Use provided config or defaults
        if config is None:
            self.config = default_config
        else:
            self.config = default_config.copy()
            # Update with provided values, keeping defaults for missing ones
            if 'training' in config:
                self.config['training'].update(config['training'])
        
        # Extract training parameters
        train_config = self.config['training']
        self.batch_size = train_config['batch_size']
        self.num_epochs = train_config['num_epochs']
        self.num_trials_per_epoch = train_config['num_trials_per_epoch']
        self.val_frequency = train_config['val_frequency']
        self.validation_trials = train_config['validation_trials']
        self.gradient_clip = train_config['gradient_clip']
        
        # Setup device
        self.device = next(model.parameters()).device
        
        # Initialize tracking metrics
        self.training_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []
        self.validation_epochs = []
        
        # Initialize dataset
        self.dataset = ngym.Dataset(
            self.env,
            env_kwargs={'dt': self.env.dt},
            batch_size=self.batch_size
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        optimizer_config = train_config['optimizer']
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=optimizer_config['weight_decay']
        )

    def train_step(self, inputs, labels):
        """Single training step with improvements."""
        self.optimizer.zero_grad()

        # Normalize inputs
        inputs = (inputs - inputs.mean()) / (inputs.std() + 1e-8)

        # Forward pass
        outputs, _ = self.model(inputs)
        
        # Reshape for loss computation
        outputs = outputs.reshape(-1, self.env.action_space.n)
        labels = labels.reshape(-1).to(self.device)

        # Calculate loss
        loss = self.criterion(outputs, labels)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()

    def validate(self):
        """Validation function."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad(), tqdm(total=self.validation_trials, desc='Validating', leave=False) as pbar:
            for _ in range(self.validation_trials):
                trial_info = self.env.new_trial()
                ob = self.env.ob
                ground_truth = trial_info['ground_truth']

                # Normalize input
                inputs = torch.from_numpy(ob[:, np.newaxis, :]).float().to(self.device)
                inputs = (inputs - inputs.mean()) / (inputs.std() + 1e-8)
                
                action_pred, _ = self.model(inputs)

                # Focus on choice period
                choice_outputs = action_pred[self.env.start_ind['choice']:self.env.end_ind['choice']]
                
                # Average predictions over choice period
                avg_choice_output = choice_outputs.mean(dim=0)
                final_choice = avg_choice_output.argmax().item() + 1

                if final_choice == ground_truth:
                    correct += 1
                total += 1
                pbar.update(1)

        accuracy = correct / total
        self.model.train()
        return accuracy

    def train(self):
        """Training function with periodic validation and batching."""
        self.model.train()

        num_batches = self.num_trials_per_epoch // self.batch_size
        all_losses = []
        accuracy_log = []

        # Create progress bar for epochs
        epoch_pbar = trange(self.num_epochs, desc='Training (epochs)')
        
        for epoch in epoch_pbar:
            epoch_loss = 0
            correct = 0
            total_samples = 0

            # Create progress bar for batches
            batch_pbar = tqdm(range(num_batches), desc=f'Epoch {epoch+1} (batches)', leave=False)
            
            for batch_idx in batch_pbar:
                # Collect batch of trials
                inputs_batch, labels_batch = [], []
                for _ in range(self.batch_size):
                    trial_info = self.env.new_trial()
                    ob, gt = self.env.ob, self.env.gt
                    inputs_batch.append(ob[:, np.newaxis, :])
                    labels_batch.append(gt)

                # Stack inputs and labels
                inputs = torch.from_numpy(np.array(inputs_batch)).float().to(self.device).squeeze()
                labels = torch.from_numpy(np.array(labels_batch)).long().to(self.device).squeeze()

                # Training step
                loss = self.train_step(inputs, labels)
                epoch_loss += loss

                # Evaluate batch performance
                with torch.no_grad():
                    action_pred, _ = self.model(inputs)
                    action_pred = action_pred.detach()
                    action_pred = action_pred[:, :, 1:]
                    preds = torch.argmax(action_pred[:, -1, :], dim=1) + 1
                    correct += (preds == labels[:, -1]).sum().item()
                    total_samples += labels.size(0)

                batch_pbar.set_postfix({'loss': f'{loss:.4f}'})

            # Calculate metrics
            epoch_accuracy = correct / total_samples
            accuracy_log.append(epoch_accuracy)
            avg_epoch_loss = epoch_loss / num_batches
            all_losses.append(avg_epoch_loss)

            # Update progress bar
            epoch_pbar.set_postfix({
                'loss': f'{avg_epoch_loss:.4f}',
                'acc': f'{epoch_accuracy:.4f}'
            })

            # Store metrics
            self.training_losses.append(avg_epoch_loss)
            self.training_accuracies.append(epoch_accuracy)
            
            # Validation
            if (epoch + 1) % self.val_frequency == 0:
                val_accuracy = self.validate()
                print(f"\nValidation Accuracy after epoch {epoch+1}: {val_accuracy:.4f}")
                self.validation_accuracies.append(val_accuracy)
                self.validation_epochs.append(epoch)

        print("\nTraining complete.")
        print("Average accuracy over all epochs:", np.mean(accuracy_log))
        print("Average loss over all epochs:", np.mean(all_losses))

        # Plot training history
        self.plot_training_history()
        return self.model

import torch
import torch.nn as nn
import numpy as np
import neurogym as ngym
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

class CompetencyTaskTrainer:
    def __init__(
        self, 
        model, 
        env, 
        config=None
    ):
        """Initialize the trainer with basic parameters.
        
        Args:
            model: The neural network model
            env: The CompetencyTask environment
            config: Configuration dictionary. If None, uses default values
        """
        self.model = model
        self.env = env
        
        # Default configuration
        default_config = {
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'num_epochs': 5,
                'num_trials_per_epoch': 200,
                'val_frequency': 1,
                'validation_trials': 100,
                'gradient_clip': 1.0,
                'optimizer': {
                    'type': 'adamw',
                    'weight_decay': 1e-4
                }
            }
        }
        
        # Use provided config or defaults
        if config is None:
            self.config = default_config
        else:
            self.config = default_config.copy()
            # Update with provided values, keeping defaults for missing ones
            if 'training' in config:
                self.config['training'].update(config['training'])
        
        # Extract training parameters
        train_config = self.config['training']
        self.batch_size = train_config['batch_size']
        self.num_epochs = train_config['num_epochs']
        self.num_trials_per_epoch = train_config['num_trials_per_epoch']
        self.val_frequency = train_config['val_frequency']
        self.validation_trials = train_config['validation_trials']
        self.gradient_clip = train_config['gradient_clip']
        
        # Setup device
        self.device = next(model.parameters()).device
        
        # Initialize tracking metrics
        self.training_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []
        self.validation_epochs = []
        
        # Initialize dataset
        self.dataset = ngym.Dataset(
            self.env,
            env_kwargs={'dt': self.env.dt},
            batch_size=self.batch_size
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        optimizer_config = train_config['optimizer']
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=optimizer_config['weight_decay']
        )

    def train_step(self, inputs, labels):
        """Single training step with improvements."""
        self.optimizer.zero_grad()

        # Normalize inputs
        inputs = (inputs - inputs.mean()) / (inputs.std() + 1e-8)

        # Forward pass
        outputs, _ = self.model(inputs)
        
        # Reshape for loss computation
        outputs = outputs.reshape(-1, self.env.action_space.n)
        labels = labels.reshape(-1).to(self.device)

        # Calculate loss
        loss = self.criterion(outputs, labels)
        
        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()

    def validate(self):
        """Validation function."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad(), tqdm(total=self.validation_trials, desc='Validating', leave=False) as pbar:
            for _ in range(self.validation_trials):
                trial_info = self.env.new_trial()
                ob = self.env.ob
                ground_truth = trial_info['ground_truth']

                # Normalize input
                inputs = torch.from_numpy(ob[:, np.newaxis, :]).float().to(self.device)
                inputs = (inputs - inputs.mean()) / (inputs.std() + 1e-8)
                
                action_pred, _ = self.model(inputs)

                # Focus on choice period
                choice_outputs = action_pred[self.env.start_ind['choice']:self.env.end_ind['choice']]
                
                # Average predictions over choice period
                avg_choice_output = choice_outputs.mean(dim=0)
                final_choice = avg_choice_output.argmax().item() + 1

                if final_choice == ground_truth:
                    correct += 1
                total += 1
                pbar.update(1)

        accuracy = correct / total
        self.model.train()
        return accuracy

    def train(self):
        """Training function with periodic validation and batching."""
        self.model.train()

        num_batches = self.num_trials_per_epoch // self.batch_size
        all_losses = []
        accuracy_log = []

        # Create progress bar for epochs
        epoch_pbar = trange(self.num_epochs, desc='Training (epochs)')
        
        for epoch in epoch_pbar:
            epoch_loss = 0
            correct = 0
            total_samples = 0

            # Create progress bar for batches
            batch_pbar = tqdm(range(num_batches), desc=f'Epoch {epoch+1} (batches)', leave=False)
            
            for batch_idx in batch_pbar:
                # Collect batch of trials
                inputs_batch, labels_batch = [], []
                for _ in range(self.batch_size):
                    trial_info = self.env.new_trial()
                    ob, gt = self.env.ob, self.env.gt
                    inputs_batch.append(ob[:, np.newaxis, :])
                    labels_batch.append(gt)

                # Stack inputs and labels
                inputs = torch.from_numpy(np.array(inputs_batch)).float().to(self.device).squeeze()
                labels = torch.from_numpy(np.array(labels_batch)).long().to(self.device).squeeze()

                # Training step
                loss = self.train_step(inputs, labels)
                epoch_loss += loss

                # Evaluate batch performance
                with torch.no_grad():
                    action_pred, _ = self.model(inputs)
                    action_pred = action_pred.detach()
                    action_pred = action_pred[:, :, 1:]
                    preds = torch.argmax(action_pred[:, -1, :], dim=1) + 1
                    correct += (preds == labels[:, -1]).sum().item()
                    total_samples += labels.size(0)

                batch_pbar.set_postfix({'loss': f'{loss:.4f}'})

            # Calculate metrics
            epoch_accuracy = correct / total_samples
            accuracy_log.append(epoch_accuracy)
            avg_epoch_loss = epoch_loss / num_batches
            all_losses.append(avg_epoch_loss)

            # Update progress bar
            epoch_pbar.set_postfix({
                'loss': f'{avg_epoch_loss:.4f}',
                'acc': f'{epoch_accuracy:.4f}'
            })

            # Store metrics
            self.training_losses.append(avg_epoch_loss)
            self.training_accuracies.append(epoch_accuracy)
            
            # Validation
            if (epoch + 1) % self.val_frequency == 0:
                val_accuracy = self.validate()
                print(f"\nValidation Accuracy after epoch {epoch+1}: {val_accuracy:.4f}")
                self.validation_accuracies.append(val_accuracy)
                self.validation_epochs.append(epoch)

        print("\nTraining complete.")
        print("Average accuracy over all epochs:", np.mean(accuracy_log))
        print("Average loss over all epochs:", np.mean(all_losses))

        # Plot training history
        self.plot_training_history()
        return self.model

    def plot_training_history(self):
        """Plot training and validation metrics history."""
        try:
            # Use only the most basic font
            plt.style.use('default')
            plt.rcParams['font.family'] = 'sans-serif'
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Plot loss with enhanced styling
            epochs = range(1, len(self.training_losses) + 1)
            ax1.plot(epochs, self.training_losses, 'b-', linewidth=2, label='Training Loss')
            ax1.set_title('Training Loss Over Time', fontsize=12, pad=10)
            ax1.set_xlabel('Epoch', fontsize=10)
            ax1.set_ylabel('Loss', fontsize=10)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(fontsize=10)
            
            # Plot accuracies with enhanced styling
            ax2.plot(epochs, self.training_accuracies, 'b-', 
                    linewidth=2, label='Training Accuracy')
            ax2.plot(np.array(self.validation_epochs) + 1, self.validation_accuracies, 
                    'r-', linewidth=2, label='Validation Accuracy')
            ax2.set_title('Training and Validation Accuracy Over Time', 
                         fontsize=12, pad=10)
            ax2.set_xlabel('Epoch', fontsize=10)
            ax2.set_ylabel('Accuracy', fontsize=10)
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend(fontsize=10)
            
            # Enhance layout and save
            plt.tight_layout(pad=3.0)
            plt.savefig('training_history.png', dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print("Training history plots have been saved to 'training_history.png'")
            
        except Exception as e:
            print(f"Error while plotting training history:", str(e))