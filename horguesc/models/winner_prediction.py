"""
Winner prediction model.
"""

import torch
from torch import nn
import logging
from .base import BaseModel

logger = logging.getLogger(__name__)

class WinnerPredictionModel(BaseModel):
    """Model for predicting race winners."""

    def __init__(self):
        """Initialize the winner prediction model."""
        super(WinnerPredictionModel, self).__init__(task_name="winner_prediction")

    def forward(self, x):
        """Forward pass through the network."""
        return x