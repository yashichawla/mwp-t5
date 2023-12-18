import nltk
import torch.nn as nn

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ContextSelectorOutput:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class ContextSelector(nn.Module):
    """
    This is the base class for all context selectors.
    It implements the basic functions that all context selectors should have.
    """

    def __init__(self):
        super(ContextSelector, self).__init__()

    def freeze(self):
        """
        This function freezes the model.

        Returns: None
        """
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """
        This function unfreezes the model.

        Returns: None
        """
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, mwps: list[str], *vargs) -> ContextSelectorOutput:
        """
        This function is the forward pass of the model.
        Args:
            mwps: The list of MWPs.
            *vargs:

        Returns: The output of the model.

        """
        output = ContextSelectorOutput()
        keywords, loss_c = self.get_keywords(mwps, *vargs)
        if loss_c is not None:
            output.loss = loss_c
        output.keywords = keywords
        return output

    def get_keywords(self, mwps: list[str], *vargs):
        """
        This function gets the keywords from the MWPs. This needs to be implemented by the child class.
        Args:
            mwps: The list of MWPs.
            *vargs:

        Returns: The keywords and the loss.

        """
        raise NotImplementedError
