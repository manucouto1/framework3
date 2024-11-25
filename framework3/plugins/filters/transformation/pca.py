from typing import Optional
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseFilter, BasePlugin
from framework3.container.container import Container
from sklearn.decomposition import PCA


@Container.bind()
class PCAPlugin(BaseFilter, BasePlugin):
    def __init__(self, n_components: int):
        super().__init__(n_components=n_components)  # Initialize the BaseFilter and BasePlugin parent classes.
        self._pca = PCA(n_components=n_components)
    
    def fit(self, x:XYData, y:Optional[XYData]) -> None:
        self._pca.fit(x)

    def predict(self, x):
        return self._pca.transform(x)