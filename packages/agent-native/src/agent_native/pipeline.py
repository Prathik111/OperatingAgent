# agent/pipeline.py

from agent_native.context import Context


class Pipeline:
    def __init__(self, layers: list):
        """
        Args:
            layers: list - Ordered list of LayerContract instances
                            e.g. [planner, executor, evaluator]
        """
        self.layers = layers

    def run(self, context: Context) -> None:
        """
        Run each layer in order against the shared context.
        Stops early if a layer marks the task as done.
        """
        for layer in self.layers:
            layer.run(context)

            if context.done:
                break

    def add_layer(self, layer) -> None:
        """
        Add a new layer to the end of the pipeline.
        """
        self.layers.append(layer)

    def insert_layer(self, index: int, layer) -> None:
        """
        Insert a layer at a specific position in the pipeline.
        """
        self.layers.insert(index, layer)

    def remove_layer(self, layer_class_name: str) -> None:
        """
        Remove a layer by its class name.

        Args:
            layer_class_name: str - e.g. "Evaluator"
        """
        self.layers = [
            layer for layer in self.layers
            if layer.__class__.__name__ != layer_class_name
        ]

    def get_layers(self) -> list:
        """
        Return the current ordered list of layers.
        """
        return self.layers