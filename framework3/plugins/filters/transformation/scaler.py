from typing import Optional
from sklearn.preprocessing import StandardScaler
from framework3.base.base_types import XYData
from framework3.base.base_clases import BaseFilter, BasePlugin
from framework3.container.container import Container

@Container.bind()
class StandardScalerPlugin(BaseFilter, BasePlugin):
    def __init__(self):
        self.scaler = StandardScaler()
    
    def fit(self, x:XYData, y:Optional[XYData]) -> None:
        self.scaler.fit(x)
    
    def predict(self, x:XYData) -> XYData:
        return self.scaler.transform(x)
