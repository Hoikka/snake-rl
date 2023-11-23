"""
Store all agents here

PyTorch implementation
"""
from replay_buffer import ReplayBuffer, ReplayBufferNumpy
import numpy as np
import time
import pickle
from collections import deque
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQLearningAgent(nn.Module):
    def __init__(
        self,
        board_size=10,
        n_frames=4,
        n_actions=3,
        buffer_size=1000,
        gamma=0.99,
        use_target_net=True,
        version="pytorch",
    ):
        super(DeepQLearningAgent, self).__init__()

        """Initialize the model

        Parameters
        ----------
        board_size : int, optional
            The board size of the environment
        frames : int, optional
            The number of frames to keep in the state
        n_actions : int, optional
            The number of actions available in the environment
        """

        self._board_size = board_size
        self._n_frames = n_frames
        self._n_actions = n_actions
        self._buffer_size = buffer_size
        self.reset_buffer()
        self._board_grid = np.arange(0, self._board_size**2).reshape(
            self._board_size, -1
        )
        self._gamma = gamma
        self._use_target_net = use_target_net
        self._version = version

        self.network = self._init_network()

        # setting the device to do stuff on
        print("Training on GPU:", torch.cuda.is_available())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)

        # RMS optimizer
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=0.0005)

        # Huber loss function
        self.loss_fn = nn.HuberLoss()

        # Initialize the target network if required
        if self._use_target_net:
            self._target_net = (
                self._init_network()
            )  # Create a separate instance for the target network
            self._target_net.to(self.device)
            self._target_net.load_state_dict(self.network.state_dict())  # Copy weights

    def _init_network(self):
        # Convolutional layers with relu activation
        conv = nn.Sequential(
            nn.Conv2d(self._n_frames, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Dense layers
        fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6400, 512),
            nn.ReLU(),
            nn.Linear(512, self._n_actions),
        )

        # Return the sequential model
        return nn.Sequential(conv, fc)

    def forward(self, x):
        # print(x.shape)
        # x = self.conv(x)
        # print(x.shape)
        # x = x.reshape(x.size(0), -1)
        # print(x.shape)
        # x = self.fc(x)
        # print(x.shape)
        # Sinde network is a sequential model, we can just call it
        return self.network(x)

    def train_agent(
        self,
        batch_size,
        num_games,
        reward_clip=True,
        # optimizer=None,
        # loss_fn=None,
        # device=None,
    ):
        total_loss = 0.0

        for i in range(num_games):
            (
                states,
                actions,
                rewards,
                next_states,
                done,
                legal_moves,
            ) = self._buffer.sample(batch_size)
            if reward_clip:
                rewards = np.sign(rewards)

            # Convert to PyTorch tensors and rearrange dimensions
            states = (
                torch.from_numpy(states).float().permute(0, 3, 1, 2)
            )  # Convert [batch, height, width, channels] to [batch, channels, height, width]
            next_states = (
                torch.from_numpy(next_states).float().permute(0, 3, 1, 2)
            )  # Same for next_states

            # Convert to PyTorch tensors and send to the appropriate device
            states = states.clone().detach().to(self.device).float()
            next_states = next_states.clone().detach().to(self.device).float()
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            done = torch.tensor(done, dtype=torch.float32).to(self.device)

            # Compute Q values for current states
            curr_Q_values = self.forward(states)

            # Gather the Q values for the actions that were taken
            curr_Q_values = curr_Q_values.gather(1, actions)

            # Compute the expected Q values
            next_Q_values = self.forward(next_states).max(1)[0]
            expected_Q_values = rewards + (
                self._gamma * next_Q_values.unsqueeze(1) * (1 - done)
            )

            # Reshape
            expected_Q_values = expected_Q_values.expand(-1, 4)

            # Compute loss
            loss = self.loss_fn(curr_Q_values, expected_Q_values.detach())

            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / num_games
        return average_loss

    def move(self, board, legal_moves, value=None):
        """Move the agent

        Parameters
        ----------
        board : np.ndarray
            The board state
        legal_moves : np.ndarray
            The legal moves available to the agent
        value : float, optional
            The value of the state, by default None

        Returns
        -------
        int
            The action to take
        """
        """
        # Convert board to tensor
        board = torch.from_numpy(board).float().unsqueeze(0)

        # Change dimensions to (batch, channels, height, width)
        board = board.permute(0, 3, 1, 2)

        # Get q-values
        q_values = self.forward(board)

        # Convert legal moves to tensor
        legal_moves = torch.from_numpy(legal_moves).float()

        # Get legal q-values
        legal_q_values = q_values * legal_moves

        # Get action
        action = torch.argmax(legal_q_values).item()

        return action
        """

        model_outputs = self._get_model_outputs(board)
        model_outputs = model_outputs.cpu().detach().numpy()

        return np.argmax(np.where(legal_moves == 1, model_outputs, -np.inf), axis=1)

    def get_gamma(self):
        """Returns the agent's gamma value

        Returns
        -------
        _gamma : float
            Agent's gamma value
        """
        return self._gamma

    def _get_model_outputs(self, board):
        """Get the model outputs

        Parameters
        ----------
        board : np.ndarray
            The board state

        Returns
        -------
        torch.Tensor
            The q-values
        """

        # Convert board to tensor
        board = torch.from_numpy(board).float().unsqueeze(0)

        # Change dimensions to (batch, channels, height, width)
        board = board.permute(0, 3, 1, 2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        board = board.to(device)

        # Get q-values
        q_values = self.forward(board)

        return q_values

    def save_model(self, file_path="", iteration=None):
        """Save the current models to disk using tensorflow's
        inbuilt save model function (saves in h5 format)
        saving weights instead of model as cannot load compiled
        model with any kind of custom object (loss or metric)

        Parameters
        ----------
        file_path : str, optional
            Path where to save the file
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        model_save_path = "{}/model_{:04d}.pt".format(file_path, iteration)
        torch.save(self.network.state_dict(), model_save_path)

        if self._use_target_net:
            target_model_save_path = "{}/model_{:04d}_target.pt".format(
                file_path, iteration
            )
            torch.save(self._target_net.state_dict(), target_model_save_path)

    def load_model(self, file_path="", iteration=None):
        """load any existing models, if available"""
        """Load models from disk using tensorflow's
        inbuilt load model function (model saved in h5 format)
        
        Parameters
        ----------
        file_path : str, optional
            Path where to find the file
        iteration : int, optional
            Iteration number the file is tagged with, if None, iteration is 0

        Raises
        ------
        FileNotFoundError
            The file is not loaded if not found and an error message is printed,
            this error does not affect the functioning of the program
        """
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        model_load_path = "{}/model_{:04d}.pt".format(file_path, iteration)
        self.network.load_state_dict(torch.load(model_load_path))

        if self._use_target_net:
            target_model_load_path = "{}/model_{:04d}_target.pt".format(
                file_path, iteration
            )
            self._target_net.load_state_dict(torch.load(target_model_load_path))
        # print("Couldn't locate models at {}, check provided path".format(file_path))

    def update_target_net(self):
        """Update the weights of the target network, which is kept
        static for a few iterations to stabilize the other network.
        This should not be updated very frequently
        """
        if self._use_target_net:
            self._target_net.load_state_dict(self.network.state_dict())
            # self._target_net.set_weights(self._model.get_weights())

    def reset_buffer(self, buffer_size=None):
        """Reset current buffer

        Parameters
        ----------
        buffer_size : int, optional
            Initialize the buffer with buffer_size, if not supplied,
            use the original value
        """
        if buffer_size is not None:
            self._buffer_size = buffer_size
        self._buffer = ReplayBufferNumpy(
            self._buffer_size, self._board_size, self._n_frames, self._n_actions
        )

    def get_buffer_size(self):
        """Get the current buffer size

        Returns
        -------
        buffer size : int
            Current size of the buffer
        """
        return self._buffer.get_current_size()

    def add_to_buffer(self, board, action, reward, next_board, done, legal_moves):
        """Add current game step to the replay buffer

        Parameters
        ----------
        board : Numpy array
            Current state of the board, can contain multiple games
        action : Numpy array or int
            Action that was taken, can contain actions for multiple games
        reward : Numpy array or int
            Reward value(s) for the current action on current states
        next_board : Numpy array
            State obtained after executing action on current state
        done : Numpy array or int
            Binary indicator for game termination
        legal_moves : Numpy array
            Binary indicators for actions which are allowed at next states
        """
        self._buffer.add_to_buffer(board, action, reward, next_board, done, legal_moves)

    def save_buffer(self, file_path="", iteration=None):
        """Save the buffer to disk

        Parameters
        ----------
        file_path : str, optional
            The location to save the buffer at
        iteration : int, optional
            Iteration number to tag the file name with, if None, iteration is 0
        """
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), "wb") as f:
            pickle.dump(self._buffer, f)

    def load_buffer(self, file_path="", iteration=None):
        """Load the buffer from disk

        Parameters
        ----------
        file_path : str, optional
            Disk location to fetch the buffer from
        iteration : int, optional
            Iteration number to use in case the file has been tagged
            with one, 0 if iteration is None

        Raises
        ------
        FileNotFoundError
            If the requested file could not be located on the disk
        """
        if iteration is not None:
            assert isinstance(iteration, int), "iteration should be an integer"
        else:
            iteration = 0
        with open("{}/buffer_{:04d}".format(file_path, iteration), "rb") as f:
            self._buffer = pickle.load(f)

    def _point_to_row_col(self, point):
        """Covert a point value to row, col value
        point value is the array index when it is flattened

        Parameters
        ----------
        point : int
            The point to convert

        Returns
        -------
        (row, col) : tuple
            Row and column values for the point
        """
        return (point // self._board_size, point % self._board_size)

    def _row_col_to_point(self, row, col):
        """Covert a (row, col) to value
        point value is the array index when it is flattened

        Parameters
        ----------
        row : int
            The row number in array
        col : int
            The column number in array
        Returns
        -------
        point : int
            point value corresponding to the row and col values
        """
        return row * self._board_size + col
