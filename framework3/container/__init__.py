from framework3.container.container import *


from framework3.plugins.storage import LocalStorage
Container.storage = LocalStorage()
Container.bind()(LocalStorage)