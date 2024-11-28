from sklearn.metrics import f1_score, precision_score, recall_score
from framework3.base.base_types import Float
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseMetric
from framework3.container.container import Container

import numpy as np

@Container.bind()
class F1(BaseMetric):
    def __init__(self, average:str='weighted'):
        super().__init__(average=average)
        self.average = average
        
    def evaluate(self, x_data:XYData, y_true: XYData, y_pred: XYData, **kwargs) -> Float|np.ndarray:
        return f1_score(y_true.value, y_pred.value, zero_division=0, average=self.average) # type: ignore

@Container.bind()
class Precission(BaseMetric):
    def __init__(self, average:str='weighted'):
        super().__init__(average=average)
        self.average = average
    def evaluate(self, x_data:XYData, y_true: XYData, y_pred: XYData, **kwargs) -> Float|np.ndarray:
        return precision_score(y_true.value, y_pred.value, zero_division=0, average=self.average, **kwargs) # type: ignore
    
@Container.bind()
class Recall(BaseMetric):
    def __init__(self, average:str='weighted'):
        super().__init__(average=average)
        self.average = average
        
    def evaluate(self, x_data:XYData, y_true: XYData, y_pred: XYData, **kwargs) -> Float|np.ndarray:
        return recall_score(y_true.value, y_pred.value, zero_division=0, average=self.average, **kwargs) # type: ignore
    