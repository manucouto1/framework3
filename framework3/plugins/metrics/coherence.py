from framework3.base.base_clases import BaseMetric
from framework3.base.base_types import Float
from framework3.base.base_types import XYData
from framework3.container.container import Container
from framework3.plugins.metrics.utils.coherence import Coherence
import numpy as np
import pandas as pd


__all__ = ["NPMI", "UMASS", "V", "UCI"]


@Container.bind()
class NPMI(BaseMetric):
    """
    Normalized Pointwise Mutual Information (NPMI) coherence metric for topic modeling evaluation.

    NPMI measures the coherence of topics based on the normalized pointwise mutual information of word pairs.

    Example:
    ```python
        >>> from framework3.plugins.metrics.coherence import NPMI
        >>> from framework3.base.base_types import XYData
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Assuming you have a DataFrame 'df' with your document-term matrix
        >>> x_data = XYData(value=df)
        >>> y_pred = np.array([['word1', 'word2', 'word3'], ['word4', 'word5', 'word6']])  # Example topics
        >>>
        >>> npmi_metric = NPMI()
        >>> score = npmi_metric.evaluate(x_data, None, y_pred, f_vocab=df.columns)
        >>> print(f"NPMI Score: {score}")
    ```
    """

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData, **kwargs
    ) -> Float | np.ndarray:
        """
        Calculate the NPMI coherence score.

        Args:
            x_data (XYData): The input data, expected to be a pandas DataFrame.
            y_true (XYData): Not used for this metric, but required by the interface.
            y_pred (XYData): The predicted topics, typically a list of lists of words.
            **kwargs (Any): Additional keyword arguments:
                - f_vocab (list): The vocabulary of the corpus.
                - topk (int): The number of top words to consider for each topic (default: 10).
                - processes (int): The number of processes to use for parallel computation (default: 1).

        Returns:
            (Float|np.ndarray): The NPMI coherence score.

        Raises:
            Exception: If x_data is not a pandas DataFrame.
        """
        f_vocab = kwargs.get("f_vocab")
        topk = kwargs.get("topk", 10)
        processes = kwargs.get("processes", 1)
        coherence = Coherence(
            f_vocab=f_vocab, topk=topk, processes=processes, measure="c_npmi"
        )
        if isinstance(x_data.value, pd.DataFrame):
            return coherence.evaluate(df=x_data.value, predicted=y_pred)
        else:
            raise Exception("x_data must be a pandas DataFrame")


@Container.bind()
class UMASS(BaseMetric):
    """
    UMass coherence metric for topic modeling evaluation.

    UMass coherence is based on document co-occurrence counts and a sliding window.

    Example:
        >>> from framework3.plugins.metrics.coherence import UMASS
        >>> from framework3.base.base_types import XYData
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Assuming you have a DataFrame 'df' with your document-term matrix
        >>> x_data = XYData(value=df)
        >>> y_pred = np.array([['word1', 'word2', 'word3'], ['word4', 'word5', 'word6']])  # Example topics
        >>>
        >>> umass_metric = UMASS()
        >>> score = umass_metric.evaluate(x_data, None, y_pred, f_vocab=df.columns)
        >>> print(f"UMass Score: {score}")
    """

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData, **kwargs
    ) -> Float | np.ndarray:
        """
        Calculate the UMass coherence score.

        Args:
            x_data (XYData): The input data, expected to be a pandas DataFrame.
            y_true (XYData): Not used for this metric, but required by the interface.
            y_pred (XYData): The predicted topics, typically a list of lists of words.
            **kwargs (Any): Additional keyword arguments:
                - f_vocab (list): The vocabulary of the corpus.
                - topk (int): The number of top words to consider for each topic (default: 10).
                - processes (int): The number of processes to use for parallel computation (default: 1).

        Returns:
            (Float|np.ndarray): The UMass coherence score.

        Raises:
            Exception: If x_data is not a pandas DataFrame.
        """
        f_vocab = kwargs.get("f_vocab")
        topk = kwargs.get("topk", 10)
        processes = kwargs.get("processes", 1)
        coherence = Coherence(
            f_vocab=f_vocab, topk=topk, processes=processes, measure="u_mass"
        )
        if isinstance(x_data.value, pd.DataFrame):
            return coherence.evaluate(df=x_data.value, predicted=y_pred)
        else:
            raise Exception("x_data must be a pandas DataFrame")


@Container.bind()
class V(BaseMetric):
    """
    V-measure coherence metric for topic modeling evaluation.

    V-measure is based on a combination of homogeneity and completeness.

    Example:
        >>> from framework3.plugins.metrics.coherence import V
        >>> from framework3.base.base_types import XYData
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Assuming you have a DataFrame 'df' with your document-term matrix
        >>> x_data = XYData(value=df)
        >>> y_pred = np.array([['word1', 'word2', 'word3'], ['word4', 'word5', 'word6']])  # Example topics
        >>>
        >>> v_metric = V()
        >>> score = v_metric.evaluate(x_data, None, y_pred, f_vocab=df.columns)
        >>> print(f"V-measure Score: {score}")
    """

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData, **kwargs
    ) -> Float | np.ndarray:
        """
        Calculate the V-measure coherence score.

        Args:
            x_data (XYData): The input data, expected to be a pandas DataFrame.
            y_true (XYData): Not used for this metric, but required by the interface.
            y_pred (XYData): The predicted topics, typically a list of lists of words.
            **kwargs (Any): Additional keyword arguments:
                - f_vocab (list): The vocabulary of the corpus.
                - topk (int): The number of top words to consider for each topic (default: 10).
                - processes (int): The number of processes to use for parallel computation (default: 1).

        Returns:
            (Float|np.ndarray): The V-measure coherence score.

        Raises:
            Exception: If x_data is not a pandas DataFrame.
        """
        f_vocab = kwargs.get("f_vocab")
        topk = kwargs.get("topk", 10)
        processes = kwargs.get("processes", 1)
        coherence = Coherence(
            f_vocab=f_vocab, topk=topk, processes=processes, measure="c_v"
        )
        if isinstance(x_data.value, pd.DataFrame):
            return coherence.evaluate(df=x_data.value, predicted=y_pred)
        else:
            raise Exception("x_data must be a pandas DataFrame")


@Container.bind()
class UCI(BaseMetric):
    """
    UCI coherence metric for topic modeling evaluation.

    UCI coherence is based on pointwise mutual information (PMI) of all word pairs in a topic.

    Example:
        >>> from framework3.plugins.metrics.coherence import UCI
        >>> from framework3.base.base_types import XYData
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # Assuming you have a DataFrame 'df' with your document-term matrix
        >>> x_data = XYData(value=df)
        >>> y_pred = np.array([['word1', 'word2', 'word3'], ['word4', 'word5', 'word6']])  # Example topics
        >>>
        >>> uci_metric = UCI()
        >>> score = uci_metric.evaluate(x_data, None, y_pred, f_vocab=df.columns)
        >>> print(f"UCI Score: {score}")
    """

    def evaluate(
        self, x_data: XYData, y_true: XYData | None, y_pred: XYData, **kwargs
    ) -> Float | np.ndarray:
        """
        Calculate the UCI coherence score.

        Args:
            x_data (XYData): The input data, expected to be a pandas DataFrame.
            y_true (XYData): Not used for this metric, but required by the interface.
            y_pred (XYData): The predicted topics, typically a list of lists of words.
            **kwargs (Any): Additional keyword arguments:
                - f_vocab (list): The vocabulary of the corpus.
                - topk (int): The number of top words to consider for each topic (default: 10).
                - processes (int): The number of processes to use for parallel computation (default: 1).

        Returns:
            (Float|np.ndarray): The UCI coherence score.

        Raises:
            Exception: If x_data is not a pandas DataFrame.
        """
        f_vocab = kwargs.get("f_vocab")
        topk = kwargs.get("topk", 10)
        processes = kwargs.get("processes", 1)
        coherence = Coherence(
            f_vocab=f_vocab, topk=topk, processes=processes, measure="c_uci"
        )
        if isinstance(x_data.value, pd.DataFrame):
            return coherence.evaluate(df=x_data.value, predicted=y_pred)
        else:
            raise Exception("x_data must be a pandas DataFrame")
