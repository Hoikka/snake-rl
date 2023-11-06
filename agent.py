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

class DeepQLearningModel(nn.Module):
    def __init__(self, board_size=10, n_frames=4, n_actions=3):
        super(DeepQLearningModel, self).__init__()

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
            nn.ReLU()
        )

        # Calculate size of flatten features after conv layers
        convw = board_size - 2
        convh = board_size - 2
        linear_input_size = convw * convh * 64

        # Dense layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_input_size, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x