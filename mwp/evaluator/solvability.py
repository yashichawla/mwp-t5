from torch import Tensor

from mwp.model.core import SolvabilityChecker
from mwp.trainer import SolvabilityCheckerTrainer


class SolvabilityEvaluator:
    def __init__(self, model: SolvabilityChecker):
        self.model = model

    def evaluate(
            self,
            mwps: list[str],
            num_equations: list[int],
            num_operands: list[int],
            operator_labels: list[Tensor],
            **kwargs
    ):
        """
        This function is used to evaluate the solvability checker model.

        Args:
            mwps: list of math word problems
            num_equations: list of number of equations in each math word problem
            num_operands: list of number of operands in each math word problem
            operator_labels: list of operator labels for each math word problem
            **kwargs: additional arguments

        Returns: dictionary containing the loss values
        """
        outputs = self.model(mwps, num_equations, num_operands, operator_labels)
        outputs_dict = outputs.__dict__
        outputs_dict["loss"] = SolvabilityCheckerTrainer.compute_loss(outputs, **kwargs)
        return outputs_dict
