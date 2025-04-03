import torch
from framework.inference.inference import Inference, DataAdapter, OutputSink
from typing import Any

class PyTorchInference(Inference):
    def __init__(self, model: torch.nn.Module, data_adapter: DataAdapter, output_sink: OutputSink):
        self.model = model
        self.data_adapter = data_adapter
        self.output_sink = output_sink

    def infer(self, input_data: Any) -> Any:
        transformed_data = self.data_adapter.transform(input_data)
        output = self.model(transformed_data)
        self.output_sink.receive({"input": input_data, "output": output})
