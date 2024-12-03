from framework3.container.container import Container


from framework3.plugins.storage import LocalStorage

Container.storage = LocalStorage()
Container.bind()(LocalStorage)
