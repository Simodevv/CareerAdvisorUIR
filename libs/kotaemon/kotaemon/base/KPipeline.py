from kotaemon.base import BaseComponent

class KPipeline(BaseComponent):
    api_key = session.get_api_key()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.components = []