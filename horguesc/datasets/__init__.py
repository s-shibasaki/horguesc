"""
Dataset modules for horguesc.
"""
from .base import BaseRacingDataset
from .winner_prediction import WinnerPredictionDataset

__all__ = [
    "BaseRacingDataset",
    "WinnerPredictionDataset",
]