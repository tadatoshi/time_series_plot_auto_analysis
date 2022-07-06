"""
This module defines stationarity categories.
"""
from enum import Enum

class Stationarity(Enum):
    """
    Enum for stationarity.
    """

    nonstationary = 'nonstationary'
    stationary = 'stationary'
