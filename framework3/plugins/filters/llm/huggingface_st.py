from framework3.base.base_types import XYData
from framework3.container.container import Container
from framework3.base.base_clases import BaseFilter, BasePlugin
from sentence_transformers import SentenceTransformer
import torch

__all__ = ["HuggingFaceSentenceTransformerPlugin"]


@Container.bind()
class HuggingFaceSentenceTransformerPlugin(BaseFilter, BasePlugin):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        self.model_name = model_name
        self._model = SentenceTransformer(self.model_name)

    def fit(self, x: XYData, y: XYData | None) -> float | None: ...

    def predict(self, x: XYData) -> XYData:
        embeddings = self._model.encode(x.value)
        return XYData.mock(torch.tensor(embeddings))
