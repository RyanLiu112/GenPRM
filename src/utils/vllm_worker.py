"""
A model worker that executes the model based on vLLM.

See documentations at docs/vllm_integration.md
"""

import argparse
import asyncio
import json
import time
from typing import List
import uuid

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.pooling_params import PoolingParams
from vllm.utils import random_uuid
from vllm.config import PoolerConfig
from transformers import AutoTokenizer
import requests

from utils.base_model_worker import BaseModelWorker
# from fastchat.serve.model_worker import (
#     logger,
#     worker_id,
# )
from fastchat.utils import get_context_length, build_logger

worker_id = str(uuid.uuid4())[:8]
time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))[2:]
logger = build_logger("model_worker", f"vllm_{time_str}.log")

app = FastAPI()


class VLLMRemoteCaller:
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    def process_prompt(self, prompt):
        if 'gemma-2' in self.args.model_path.lower():
            eos_token = '<end_of_turn>'
        else:
            eos_token = self.tokenizer.eos_token

        if prompt.endswith(f"{eos_token}\n"):
            prompt = prompt[:-len(f"{eos_token}\n")]
        elif prompt.endswith(eos_token):
            prompt = prompt[:-len(eos_token)]
        return prompt

    def generate_fastchat(self, messages, n, temperature):
        if self.args.apply_chat_template:
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt = self.process_prompt(prompt)
        else:
            prompt = messages

        headers = {"User-Agent": "FastChat Client"}
        gen_params = {
            "model": self.args.model_path,
            "prompt": prompt,
            "temperature": temperature,
            "n": n,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "max_new_tokens": self.args.max_new_tokens,
            "echo": False,
        }
        try:
            response = requests.post(self.args.worker_addr + "/worker_generate", headers=headers, json=gen_params, stream=True, verify=False)
            results = response.json()
        except Exception as e:
            print(f'Error in _generate_fastchat: {e}')

        return results["text"], results["finish_reason"]


class VLLMWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        llm_engine: AsyncLLMEngine,
        conv_template: str,
        is_embedding: bool,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
            is_embedding,
        )

        # self.encode_semaphore = asyncio.Semaphore(1)

        logger.info(f"Loading the model {self.model_names} on worker {worker_id}, worker type: vLLM worker...")
        self.tokenizer = llm_engine.engine.tokenizer.tokenizer
        self.context_len = get_context_length(llm_engine.engine.model_config.hf_config)

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):
        self.call_ct += 1

        context = params.pop("prompt")
        n = params.get("n", 1)
        request_id = params.pop("request_id")
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        top_k = params.get("top_k", -1.0)
        presence_penalty = float(params.get("presence_penalty", 0.0))
        frequency_penalty = float(params.get("frequency_penalty", 0.0))
        max_new_tokens = params.get("max_new_tokens", 256)
        stop_str = params.get("stop", None)
        stop_token_ids = params.get("stop_token_ids", None) or []
        if self.tokenizer.eos_token_id is not None:
            stop_token_ids.append(self.tokenizer.eos_token_id)
        echo = params.get("echo", False)
        use_beam_search = params.get("use_beam_search", False)
        best_of = params.get("best_of", None)
        include_stop_str_in_output = params.get("include_stop_str_in_output", False)

        # Handle stop_str
        stop = set()
        if isinstance(stop_str, str) and stop_str != "":
            stop.add(stop_str)
        elif isinstance(stop_str, list) and stop_str != []:
            stop.update(stop_str)

        # for tid in stop_token_ids:
        #     if tid is not None:
        #         stop.add(self.tokenizer.decode(tid))

        # make sampling params in vllm
        top_p = max(top_p, 1e-5)
        if temperature <= 1e-5:
            top_p = 1.0
        sampling_params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            # use_beam_search=use_beam_search,
            stop=list(stop),
            stop_token_ids=stop_token_ids,
            max_tokens=max_new_tokens,
            top_k=top_k,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            best_of=best_of,
            logprobs=1,
            include_stop_str_in_output=include_stop_str_in_output,
        )
        results_generator = engine.generate(context, sampling_params, request_id)

        async for request_output in results_generator:
            # prompt = request_output.prompt
            # if echo:
            #     text_outputs = [prompt + output.text for output in request_output.outputs]
            # else:
            text_outputs = [output.text for output in request_output.outputs]
            # text_outputs = " ".join(text_outputs)
            # Note: usage is not supported yet
            prompt_tokens = len(request_output.prompt_token_ids)
            completion_tokens = sum(len(output.token_ids) for output in request_output.outputs)
            ret = {
                "text": text_outputs,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "cumulative_logprob": [output.cumulative_logprob for output in request_output.outputs],
                "output_token_len": [len(output.token_ids) for output in request_output.outputs],
                "finish_reason": [output.finish_reason for output in request_output.outputs],
            }
            yield (json.dumps(ret) + "\0").encode()

    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        return json.loads(x[:-1].decode())

    async def encode_stream(self, params):
        # async with self.encode_semaphore:  # 获取 Semaphore, 确保按顺序执行
        self.call_ct += 1

        context = params.pop("prompt")
        n = params.get("n", 1)
        request_id = params.pop("request_id")

        pooling_params = PoolingParams()
        results_generator = engine.encode(context, pooling_params, request_id)

        async for request_output in results_generator:
            embedding_outputs = request_output.outputs.embedding
            prompt_tokens = len(request_output.prompt_token_ids)
            ret = {
                "embedding": embedding_outputs,
                "error_code": 0,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                },
                "finished": request_output.finished,
            }
            yield (json.dumps(ret) + "\0").encode()

    async def encode(self, params):
        # async with self.encode_semaphore:  # 获取 Semaphore, 确保按顺序执行
        async for x in self.encode_stream(params):
            pass
        return json.loads(x[:-1].decode())


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(request_id):
    async def abort_request() -> None:
        await engine.abort(request_id)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    output = await worker.generate(params)
    release_worker_semaphore()
    await engine.abort(request_id)
    return JSONResponse(output)


@app.post("/worker_encode_stream")
async def api_encode_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    generator = worker.encode_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_encode")
async def api_encode(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = random_uuid()
    params["request_id"] = request_id
    output = await worker.encode(params)
    release_worker_semaphore()
    await engine.abort(request_id)
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str, default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-template", type=str, default=None, help="Conversation prompt template.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_false",
        default=True,
        help="Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="The ratio (between 0 and 1) of GPU memory to"
        "reserve for the model weights, activations, and KV cache. Higher"
        "values will increase the KV cache size and thus improve the model's"
        "throughput. However, if the value is too high, it may cause out-of-"
        "memory (OOM) errors.",
    )
    parser.add_argument(
        '--swap_space',
        type=float,
        default=4,
        help='CPU swap space size (GiB) per GPU.'
    )
    parser.add_argument("--max_model_length", type=int, default=0)
    parser.add_argument("--max_num_sequences", type=int, default=0)
    parser.add_argument("--enable_chunked_prefill", action="store_true")
    parser.add_argument("--is_embedding", action="store_true")

    if 'gemma-2' in parser.parse_args().model_path.lower():
        try:
            import sglang as sgl
            llm = sgl.Engine(model_path="")
        except Exception as e:
            pass
        raise ValueError(f"Use SGlang for Gemma-2 Series Models")
        # print(f"Use SGlang for Gemma-2 Series Models")
        # import sglang as sgl
        # import asyncio
        # llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")
    else:
        parser = AsyncEngineArgs.add_cli_args(parser)
        args = parser.parse_args()
        if args.model_path:
            args.model = args.model_path
        if args.num_gpus > 1:
            args.tensor_parallel_size = args.num_gpus
        if args.max_model_length > 0:
            args.max_model_len = args.max_model_length
        if args.max_num_sequences > 0:
            args.max_num_seqs = args.max_num_sequences

        if 'ministral' in args.model_path.lower() or 'mistral' in args.model_path.lower():
            args.tokenizer_mode = 'mistral'
            args.config_format = 'mistral'
            args.load_format = 'mistral'

        if args.is_embedding:
            # args.limit_worker_concurrency = 1
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            if 'math-shepherd' in args.model_path.lower():
                GOOD_TOKEN, BAD_TOKEN = '+', '-'
                STEP_TAG = 'ки'
                # pooling_type = 'STEP'
                # step_tag_id = tokenizer.encode(f"{STEP_TAG}")[-1]  # 12902
                # returned_token_ids = tokenizer.encode(f"{GOOD_TOKEN} {BAD_TOKEN}")[1:]  # [648, 387]
                pooling_type = 'ALL'
                step_tag_id = None
                returned_token_ids = None
            elif 'llama' in args.model_path.lower():
                pooling_type = 'ALL'
                step_tag_id = None
                returned_token_ids = None
            elif 'qwen' in args.model_path.lower():
                pooling_type = 'ALL'
                step_tag_id = None
                returned_token_ids = None
            else:
                raise ValueError(f"Invalid model path: {args.model_path} for embedding model")
            pooler_config = PoolerConfig(
                pooling_type=pooling_type,
                normalize=False,
                softmax=False,
                step_tag_id=step_tag_id,
                returned_token_ids=returned_token_ids,
            )
            args.override_pooler_config = pooler_config

        engine_args = AsyncEngineArgs.from_cli_args(args)
        engine = AsyncLLMEngine.from_engine_args(engine_args)

    worker = VLLMWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_path,
        args.model_names,
        args.limit_worker_concurrency,
        args.no_register,
        engine,
        args.conv_template,
        args.is_embedding,
    )
    print(args)
    print('*' * 50)
    print(engine_args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
