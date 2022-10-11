#
# sortin-hat
# Pipeline
# ML Models
#


from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class Model(ABC):
    """Abstract ML Model interface implemented by all models."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path):
        """Restore a Model from the file at the given path."""
        pass

    @abstractmethod
    def save(self, path: Path):
        """Restore a Model from the file at the given path."""
        pass

    @abstractmethod
    def fit(self, features: NDArray[np.float_], labels: NDArray[np.float_]):
        """Fit the model to the given training batch features & labels."""
        pass

    @abstractmethod
    def predict(self, features: NDArray[np.float_]) -> NDArray[np.float_]:
        """Make a batch of predictions using the given input features."""
        pass
