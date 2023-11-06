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


class DeepQLearningAgent(nn.Module):
    def __init__(self, board_size=10, n_frames=4, n_actions=3):
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

        # Convolutional layers with relu activation
        self.conv = nn.Sequential(
            nn.Conv2d(n_frames, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Dense layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6400, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        return x

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

        # Get q-values
        q_values = self.forward(board)

        return q_values
