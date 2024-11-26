from typing import Optional
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseFilter
from framework3.container.container import Container
from sklearn.decomposition import PCA


@Container.bind()
class PCAPlugin(BaseFilter):
    def __init__(self, n_components: int):
        super().__init__(n_components=n_components)  # Initialize the BaseFilter and BasePlugin parent classes.
        self._pca = PCA(n_components=n_components)
    
    def fit(self, x:XYData, y:Optional[XYData]) -> None:
        self._pca.fit(x.value)

    def predict(self, x:XYData):
        return self._pca.transform(x.value)