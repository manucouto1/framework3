
from framework3.base.base_clases import BaseMetric
from framework3.base.base_types import Float
from framework3.base.base_types import XYData
from framework3.container.container import Container
from framework3.plugins.metrics.utils.coherence import Coherence
import numpy as np
import pandas as pd

@Container.bind()
class NPMI(BaseMetric):
    def evaluate(self,x_data:XYData, y_true: XYData, y_pred: XYData, **kwargs) -> Float|np.ndarray:
        # Implementation of UMASS metric goes here
        f_vocab = kwargs.get('f_vocab')
        topk = kwargs.get('topk', 10)
        processes = kwargs.get('processes', 1)
        coherence = Coherence(f_vocab=f_vocab, topk=topk, processes=processes, measure='c_npmi')
        if isinstance(x_data, pd.DataFrame):
            return coherence.evaluate(df=x_data, predicted=y_pred)
        else:
            raise Exception('x_data must be a pandas DataFrame')

@Container.bind()
class UMASS(BaseMetric):
    def evaluate(self,x_data:XYData, y_true: XYData, y_pred: XYData, **kwargs) -> Float|np.ndarray:
        # Implementation of UMASS metric goes here
        f_vocab = kwargs.get('f_vocab')
        topk = kwargs.get('topk', 10)
        processes = kwargs.get('processes', 1)
        coherence = Coherence(f_vocab=f_vocab, topk=topk, processes=processes, measure='u_mass')
        if isinstance(x_data, pd.DataFrame):
            return coherence.evaluate(df=x_data, predicted=y_pred)
        else:
            raise Exception('x_data must be a pandas DataFrame')

@Container.bind()     
class V(BaseMetric):
    def evaluate(self,x_data:XYData, y_true: XYData, y_pred: XYData, **kwargs) -> Float|np.ndarray:
        # Implementation of UMASS metric goes here
        f_vocab = kwargs.get('f_vocab')
        topk = kwargs.get('topk', 10)
        processes = kwargs.get('processes', 1)
        coherence = Coherence(f_vocab=f_vocab, topk=topk, processes=processes, measure='c_v')
        if isinstance(x_data, pd.DataFrame):
            return coherence.evaluate(df=x_data, predicted=y_pred)
        else:
            raise Exception('x_data must be a pandas DataFrame')

@Container.bind()   
class UCI(BaseMetric):
    def evaluate(self,x_data:XYData, y_true: XYData, y_pred: XYData, **kwargs) -> Float|np.ndarray:
        # Implementation of UMASS metric goes here
        f_vocab = kwargs.get('f_vocab')
        topk = kwargs.get('topk', 10)
        processes = kwargs.get('processes', 1)
        coherence = Coherence(f_vocab=f_vocab, topk=topk, processes=processes, measure='c_uci')
        if isinstance(x_data, pd.DataFrame):
            return coherence.evaluate(df=x_data, predicted=y_pred)
        else:
            raise Exception('x_data must be a pandas DataFrame')