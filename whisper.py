from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from torch.nn.attention import SDPBackend, sdpa_kernel
import torch
import torch._dynamo as dynamo
dynamo.config.ignore_logger_methods.add("warning_once")
torch.set_float32_matmul_precision("high")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
).to(device)

model.generation_config.cache_implementation = "static"
model.generation_config.max_new_tokens = 256
model.forward = torch.compile(
    model.forward, mode="reduce-overhead", fullgraph=True)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

for _ in tqdm(range(2), desc="Warm-up step"):
    with sdpa_kernel(SDPBackend.MATH):
        result = pipe("warmup.mp3", return_timestamps=True, chunk_length_s=30, generate_kwargs={
                      "task": "transcribe", "language": "english", "min_new_tokens": 256, "max_new_tokens": 256})
        print(result)
