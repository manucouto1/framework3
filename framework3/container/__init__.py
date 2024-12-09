from framework3.container.container import Container
from framework3.plugins.storage import LocalStorage
from framework3.plugins.ingestion import DatasetManager


Container.storage = LocalStorage()
Container.ds = DatasetManager()
Container.bind()(LocalStorage)
