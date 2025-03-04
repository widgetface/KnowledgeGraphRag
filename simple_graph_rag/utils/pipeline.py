from typing import List
from haystack import component
from haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.generators.ollama import OllamaGenerator


@component
class ReviewDatasetPipeline:
    def __init__(self, model: str, template: str, url: str = "http://localhost:11434"):
        pipe = Pipeline()
        pipe.add_component("prompt_builder", PromptBuilder(template=template))
        pipe.add_component("llm", OllamaGenerator(model=model, url=url))
        pipe.connect("prompt_builder", "llm")
        self.pipeline = pipe

    @component.output_types(data=List)
    def run(
        self,
        query: str,
        text: str,
    ):
        response = self.pipeline.run({"prompt_builder": {"query": query, "text": text}})
        return response["llm"]["replies"]
