from sklearn.metrics import completeness_score, normalized_mutual_info_score, adjusted_rand_score, silhouette_score, calinski_harabasz_score, homogeneity_score, completeness_score
from framework3.base.base_types import Float
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseMetric
from framework3.container.container import Container
from typing import Any

import numpy as np

@Container.bind()
class NMI(BaseMetric):
    def evaluate(self,x_data:XYData, y_true: Any, y_pred: Any, **kwargs) -> Float|np.ndarray:
        return normalized_mutual_info_score(y_true, y_pred, **kwargs)

@Container.bind()
class ARI(BaseMetric):
    def evaluate(self,x_data:XYData, y_true: Any, y_pred: Any, **kwargs) -> Float|np.ndarray:
        return adjusted_rand_score(y_true, y_pred, **kwargs)

@Container.bind()
class Silhouette(BaseMetric):
    def evaluate(self,x_data:XYData, y_true: Any, y_pred: Any, **kwargs) -> Float|np.ndarray:
        return silhouette_score(x_data, y_pred, **kwargs)

@Container.bind()
class CalinskiHarabasz(BaseMetric):
    def evaluate(self,x_data:XYData, y_true: Any, y_pred: Any, **kwargs) -> Float|np.ndarray:
        return calinski_harabasz_score(x_data, y_pred, **kwargs)
    
@Container.bind()
class Homogeneity(BaseMetric):
    def evaluate(self,x_data:XYData, y_true: Any, y_pred: Any, **kwargs) -> Float|np.ndarray:
        return homogeneity_score(y_true, y_pred, **kwargs)
    
@Container.bind()
class Completness(BaseMetric):
    def evaluate(self,x_data:XYData, y_true: Any, y_pred: Any, **kwargs) -> Float|np.ndarray:
        return completeness_score(y_true, y_pred, **kwargs)
