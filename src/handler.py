import os
import runpod
from utils import JobInput

_vllm_engine = None
_openai_engine = None

def init_engines():
    global _vllm_engine, _openai_engine
    if _vllm_engine is None:
        from engine import vLLMEngine, OpenAIvLLMEngine

        _vllm_engine = vLLMEngine()
        _openai_engine = OpenAIvLLMEngine(_vllm_engine)

async def handler(job):
    init_engines()

    job_input = JobInput(job["input"])
    engine = _openai_engine if job_input.openai_route else _vllm_engine

    async for batch in engine.generate(job_input):
        yield batch
        
def concurrency_modifier(_current_concurrency: int):
    return int(os.getenv("MAX_CONCURRENCY", "1"))

if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": concurrency_modifier,
            "return_aggregate_stream": True,
        }
    )
