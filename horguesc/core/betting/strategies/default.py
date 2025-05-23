import torch
import logging
from typing import Dict, Any, Optional
import numpy as np
from horguesc.core.betting.strategies.base import BaseStrategy
from horguesc.core.betting.strategies.kelly_based import KellyBasedStrategy
from horguesc.core.betting.strategies.ranked_value import RankedValueStrategy

logger = logging.getLogger(__name__)


default_strategies = {
    'KellyBased': KellyBasedStrategy,
    'RankedValue': RankedValueStrategy
}