class Context:
    def __init__(self, task: str):
        self.task = task
        self.plan = None
        self.observations = []
        self.result = None
        self.done = False
        self.history = []
        self.metrics = []
    
    def add_metrics(self, layer: str, metrics: dict):
        self.metrics.append({"layer": layer, **metrics})