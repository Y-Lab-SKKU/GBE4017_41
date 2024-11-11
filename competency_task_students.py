import numpy as np
from neurogym import spaces, TrialEnv
import neurogym as ngym


class CompetencyTask(ngym.TrialEnv):
    """Two-phase competency task with improved fish movement based on MATLAB implementation."""
    def __init__(self, dt=100, timing=None):
        super().__init__(dt=dt)
        
        # Initialize rewards dictionary
        self.rewards = {
            'correct': +1.,
            'fail': 0.,
            'abort': -0.1
        }
        
        # Screen layout parameters
        self.screen_bounds = [-2, 2]  # Screen boundaries
        self.screen_width = self.screen_bounds[1] - self.screen_bounds[0]
        self.screen_height = 4  # -2 to 2 vertical range
        
        # Eel parameters
        self.eel_speed = 0.02
        self.eel_size = 0.1
        self.eel_wiggle = 0.01
        
        # Fish parameters
        self.n_fish = 12
        self.fish_size = 0.05
        self.min_separation = (self.eel_size + self.fish_size) / 2
        
        # Movement speeds
        self.fish_speed_slow = 0.002
        self.fish_speed_fast = 0.05
        
        # Potential field parameters
        self.potential_levels = [0.4, 0.6, 0.8, 1.0]
        self.outer_radius_factor = 0.5
        
        # Movement parameters
        self.eel_base_weight = 10.0
        self.eel_exp_scale = 1.5
        self.momentum_weight = 0.4
        self.direction_penalty = 0.3
        self.noise_factor = 0.3
        
        if timing == None:
            # Timing
            self.timing = {
                'fixation': 100,
                'eel1_observe': 100,
                'iti1': 20,
                'eel2_observe': 100,
                'iti2': 20,
                'choice':50,
            }
        else: 
            self.timing.update(timing)

        
        # Observation space setup
        total_dims = (
            1 +                     # Fixation point
            2 +                     # Eel 1 position
            (self.n_fish * 2) +    # Eel 1 fish positions
            2 +                     # Eel 2 position
            (self.n_fish * 2)      # Eel 2 fish positions
        )
        
        # Set up observation space
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(total_dims,), dtype=np.float32,
            name={
                'fixation': 0,
                'eel1': [1, 2],
                'eel1_fish': list(range(3, 3 + self.n_fish * 2)),
                'eel2': [3 + self.n_fish * 2, 4 + self.n_fish * 2],
                'eel2_fish': list(range(5 + self.n_fish * 2, total_dims))
            }
        )
        
        # Action space: fixate (0), choose eel1 (1), choose eel2 (2)
        self.action_space = spaces.Discrete(3)
        self.action_space.name = {'fixation': 0, 'choice': [1, 2]}


    def _new_trial(self, **kwargs):
        """Initialize a new trial with two eels and their fish."""
        # Trial periods
        self.add_period(['fixation', 'eel1_observe', 'iti1',
                        'eel2_observe', 'iti2', 'choice'])
        
        # Trial info
        trial = {'first_side': self.rng.choice(['left', 'right'])}
        trial.update(kwargs)
        
        # Assign potential field sizes
        potentials = self.rng.choice(self.potential_levels, 2, replace=False)
        trial['eel1_potential'] = potentials[0]
        trial['eel2_potential'] = potentials[1]
        
        # Calculate outer radii
        trial['eel1_outer_radius'] = trial['eel1_potential'] * self.outer_radius_factor
        trial['eel2_outer_radius'] = trial['eel2_potential'] * self.outer_radius_factor
        
        # Initialize positions based on first_side
        if trial['first_side'] == 'left':
            self.eel1_pos = np.array([-1.0, 0.0])
            self.eel2_pos = np.array([1.0, 0.0])
        else:
            self.eel1_pos = np.array([1.0, 0.0])
            self.eel2_pos = np.array([-1.0, 0.0])
            
        # Initialize fish positions
        self.fish1_pos = self._initialize_fish(
            self.eel1_pos, trial['eel1_potential'], trial['eel1_outer_radius'])
        self.fish2_pos = self._initialize_fish(
            self.eel2_pos, trial['eel2_potential'], trial['eel2_outer_radius'])
            
        # Initialize previous positions for momentum
        self.prev_fish1_pos = self.fish1_pos.copy()
        self.prev_fish2_pos = self.fish2_pos.copy()
        
        # Set ground truth (larger potential field is correct)
        ground_truth = 0 if trial['eel1_potential'] > trial['eel2_potential'] else 1
        trial['ground_truth'] = ground_truth
        
        # Set observations
        self.add_ob(-1, period=['fixation', 'eel1_observe', 'iti1',
                              'eel2_observe', 'iti2', 'choice'], where='fixation')
        
        # Set ground truth
        self.set_groundtruth(ground_truth, period='choice')
         # Assign potential field sizes
        potentials = self.rng.choice(self.potential_levels, 2, replace=False)
        trial['eel1_potential'] = potentials[0]
        trial['eel2_potential'] = potentials[1]

        # Set ground truth (larger potential field is correct)
        ground_truth = 1 if trial['eel1_potential'] > trial['eel2_potential'] else 2
        trial['ground_truth'] = ground_truth

        # Set ground truth for the entire trial duration
        self.set_groundtruth(ground_truth, period='choice')
        return trial
        

    def _initialize_fish(self, eel_pos, potential_radius, outer_radius):
        """Initialize fish positions with natural distribution around eel."""
        positions = []
        n_inner = self.n_fish // 3
        n_outer = self.n_fish - n_inner

        # Calculate the radius range for the inner and outer fish
        inner_radius_range = (0.2 * potential_radius, potential_radius)
        outer_radius_range = (potential_radius, outer_radius)

        # Place fish inside potential field
        for i in range(n_inner):
            angle = (i / n_inner) * 2 * np.pi + self.rng.uniform(0, 0.5)
            radius = self.rng.uniform(*inner_radius_range)
            pos = eel_pos + radius * np.array([np.cos(angle), np.sin(angle)])
            positions.append(pos)

        # Place fish in outer radius
        for i in range(n_outer):
            angle = (i / n_outer) * 2 * np.pi + self.rng.uniform(0, 0.5)
            radius = self.rng.uniform(*outer_radius_range)
            pos = eel_pos + radius * np.array([np.cos(angle), np.sin(angle)])
            positions.append(pos)

        return np.array(positions)
    
    def _update_eel_position(self, eel_pos, is_first_eel):
        """Update eel position while keeping it on its designated side.
        
        Args:
            eel_pos: Current position of the eel
            is_first_eel: Boolean indicating if this is the first eel (left side if first_side='left')
        """
        # Calculate center point for this eel based on its side
        if self.trial['first_side'] == 'left':
            is_left_eel = is_first_eel
        else:
            is_left_eel = not is_first_eel
        
        # Set side-specific boundaries and center point
        if is_left_eel:
            side_center = np.array([-1.0, 0.0])
            x_bounds = [self.screen_bounds[0], 0.0]  # Left half of screen
        else:
            side_center = np.array([1.0, 0.0])
            x_bounds = [0.0, self.screen_bounds[1]]  # Right half of screen
        
        # Calculate random movement
        movement = self.rng.normal(0, self.eel_wiggle, size=2)
        
        # Add attraction to side's center point
        to_center = side_center - eel_pos
        center_attraction = 0.02 * to_center  # Slightly stronger center attraction
        
        # Combine movements
        new_pos = eel_pos + movement + center_attraction
        
        # Keep within side bounds
        new_pos[0] = np.clip(new_pos[0], x_bounds[0], x_bounds[1])
        new_pos[1] = np.clip(new_pos[1], -self.screen_height/2, self.screen_height/2)
        
        return new_pos
    

    def _update_fish_positions(self, fish_pos, prev_fish_pos, eel_pos, potential_radius, outer_radius):
        """Update fish positions with free movement.
        
        Fish behavior:
        1. Move freely both inside and outside potential radius
        2. Only constrained by outer radius
        3. Speed changes based on position relative to potential radius
        """
        updated_pos = []
        
        for i, pos in enumerate(fish_pos):
            # Get current direction
            prev_pos = prev_fish_pos[i]
            current_direction = pos - prev_pos
            current_direction_norm = np.linalg.norm(current_direction)
            
            if current_direction_norm > 0:
                current_direction = current_direction / current_direction_norm
            else:
                # If no movement, pick random direction
                angle = self.rng.uniform(0, 2 * np.pi)
                current_direction = np.array([np.cos(angle), np.sin(angle)])
                
            # Distance to eel
            to_eel = eel_pos - pos
            dist_to_eel = np.linalg.norm(to_eel)
            
            # Just wander with speed changes
            speed = self.fish_speed_slow if dist_to_eel <= potential_radius else self.fish_speed_fast
            
            # Random direction change
            angle_change = self.rng.uniform(-0.25, 0.25)
            cos_theta = np.cos(angle_change)
            sin_theta = np.sin(angle_change)
            new_direction = np.array([
                current_direction[0] * cos_theta - current_direction[1] * sin_theta,
                current_direction[0] * sin_theta + current_direction[1] * cos_theta
            ])
            
            # Only influence direction if beyond outer radius
            if dist_to_eel > outer_radius:
                to_eel_norm = to_eel / dist_to_eel
                new_direction = 0.3 * new_direction + 0.7 * to_eel_norm
                new_direction = new_direction / np.linalg.norm(new_direction)
                
            # Move fish
            new_pos = pos + new_direction * speed
            
            # Add very small random movement for natural feel
            new_pos += self.rng.normal(0, speed * 0.1, size=2)
            
            # Handle screen boundaries
            for j in range(2):
                if new_pos[j] < self.screen_bounds[0]:
                    new_pos[j] = self.screen_bounds[0]
                    new_direction[j] *= -1
                elif new_pos[j] > self.screen_bounds[1]:
                    new_pos[j] = self.screen_bounds[1]
                    new_direction[j] *= -1
            
            updated_pos.append(new_pos)
        
        return np.array(updated_pos)

    def _get_observation(self):
        """Return current observation based on task phase."""
        obs = np.zeros(self.observation_space.shape[0])
        
        # Fixation point always visible
        obs[0] = 1
        
        if self.in_period('eel1_observe'):
            # Show eel1 and its fish
            obs[1:3] = self.eel1_pos
            fish_flat = self.fish1_pos.flatten()
            obs[3:3 + len(fish_flat)] = fish_flat
            
        elif self.in_period('eel2_observe'):
            # Show eel2 and its fish
            idx = 3 + self.n_fish * 2
            obs[idx:idx + 2] = self.eel2_pos
            fish_flat = self.fish2_pos.flatten()
            obs[idx + 2:idx + 2 + len(fish_flat)] = fish_flat
            
        return obs

    def _step(self, action):
        """Process one action and update positions."""
        new_trial = False
        reward = 0

        # Update positions in relevant periods
        if self.in_period('eel1_observe'):
            self.eel1_pos = self._update_eel_position(self.eel1_pos, is_first_eel=True)
            outer_radius = self.trial['eel1_potential'] * 2  # Outer radius is 2x potential
            self.fish1_pos = self._update_fish_positions(
                self.fish1_pos, self.prev_fish1_pos, 
                self.eel1_pos, 
                self.trial['eel1_potential'],
                outer_radius
            )

        elif self.in_period('eel2_observe'):
            self.eel2_pos = self._update_eel_position(self.eel2_pos, is_first_eel=False)
            outer_radius = self.trial['eel2_potential'] * 2  # Outer radius is 2x potential
            self.fish2_pos = self._update_fish_positions(
                self.fish2_pos, self.prev_fish2_pos,
                self.eel2_pos,
                self.trial['eel2_potential'],
                outer_radius
            )

        # Handle choice period
        if self.in_period('choice'):
            if action != 0:
                new_trial = True
                if action == self.gt_now:
                    reward = self.rewards['correct']
                    self.performance = 1
                else:
                    reward = self.rewards['fail']
                    self.performance = 0

        return self._get_observation(), reward, False, {'new_trial': new_trial}