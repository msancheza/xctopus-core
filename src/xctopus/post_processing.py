import logging
import torch
from typing import Dict, List, Any, Optional
from .settings import PP_HIGH_THRESHOLD, PP_LOW_THRESHOLD, PP_ETA


logger = logging.getLogger(__name__)

from dataclasses import dataclass

@dataclass
class Feedback:
    status: str              # "OK" or "NEW_BUFFER"
    delta_mass: float = 0.0  # change to node mass
    delta_variance: float = 0.0 # change to node variance

class PostProcessor:
    """
    Quality Evaluator and Feedback Dispatcher.
    Analyzes Transformer output and decides how to adjust Knowledge Nodes.
    """
    def __init__(self):
        # Configurable thresholds (from settings)
        self.h_threshold = PP_HIGH_THRESHOLD
        self.l_threshold = PP_LOW_THRESHOLD
        self.eta = PP_ETA # Learning rate for variance

    def evaluate(self, enriched_response: Dict[str, Any]) -> Feedback:
        """
        Calculates feedback deltas based on response confidence.
        Returns a Feedback object that the Orchestrator will apply.
        """
        confidence = enriched_response.get("confidence", 0.0)
        
        # If confidence is very low, we suggest creating a new buffer
        # because the current node does not seem competent.
        if confidence < self.l_threshold:
            logger.debug(f"Very low confidence ({confidence:.2f}). Suggesting NEW_BUFFER.")
            return Feedback(status="NEW_BUFFER")
            
        # If acceptable, we calculate reinforcement/penalty
        delta_mass = 0.0
        delta_variance = 0.0
        
        if confidence >= self.h_threshold:
            # Reinforce: increase mass and reduce variance
            delta_mass = 1.0 # +1 successful embedding
            delta_variance = -self.eta * confidence # reduce variance
            logger.debug(f"High confidence ({confidence:.2f}). Reinforcing node.")
        else:
            # Middle zone: maintain stability or slight penalty
            # (confidence between low and high)
            delta_mass = 0.5 # Partial contribution
            delta_variance = self.eta * (1 - confidence) # increase variance (uncertainty)
            logger.debug(f"Mid confidence ({confidence:.2f}). Soft adjustment.")
            
        return Feedback(
            status="OK",
            delta_mass=delta_mass,
            delta_variance=delta_variance
        )