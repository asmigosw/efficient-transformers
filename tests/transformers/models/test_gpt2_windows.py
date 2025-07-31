from transformers import AutoModelForCausalLM
from QEfficient.utils import hf_download
from QEfficient.utils.constants import Constants
from QEfficient.utils.run_utils import ApiRunner
from QEfficient.utils._utils import load_hf_tokenizer
from QEfficient.transformers.models.modeling_auto import QEFFAutoModelForCausalLM

def load_causal_lm_model(model_config):
    """
    Function to load model from huggingface and transform to KV model
    --------

    :model_config: Dict

    :return model_hf, params
    """
    model_path = hf_download(
        repo_id=model_config["model_name"],
        ignore_patterns=["*.onnx", "*.ot", "*.md", "*.tflite", "*.pdf", "*.h5", "*.msgpack"],
    )
    model_hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_cache=True,
        attn_implementation="eager",
        low_cpu_mem_usage=False,
    )  # Run models for single layers only
    params = sum(p.numel() for p in model_hf.parameters())
    model_hf.eval()
    return model_hf, params

model_name = "gpt2"

model_config = {"model_name": model_name}

model_hf, _ = load_causal_lm_model(model_config)

tokenizer = load_hf_tokenizer(pretrained_model_name_or_path=model_name)
config = model_hf.config
batch_size = len(Constants.INPUT_STR)
api_runner = ApiRunner(
    batch_size,
    tokenizer,
    config,
    Constants.INPUT_STR,
    Constants.PROMPT_LEN,
    Constants.CTX_LEN,
)

# pytorch_hf_tokens = api_runner.run_hf_model_on_pytorch(model_hf)
# print("Pytorch HF tokens:", pytorch_hf_tokens)

qeff_model = QEFFAutoModelForCausalLM(model_hf, pretrained_model_name_or_path=model_name)

# pytorch_kv_tokens = api_runner.run_kv_model_on_pytorch(qeff_model.model)
# print("Pytorch KV tokens:", pytorch_kv_tokens)

qpc_path = qeff_model.export()
print("qpc_path: ", qpc_path)

# qpc_path = qeff_model.compile(
#     prefill_seq_len=Constants.PROMPT_LEN,
#     ctx_len=Constants.CTX_LEN,
#     num_cores=16,
#     mxfp6_matmul=False,
#     mxint8_kv_cache=False,
#     num_devices=1,
#     mos=1,
#     aic_enable_depth_first=True,
#     num_speculative_tokens=None,
# )
# print("Compiled Successfully at path: ", qpc_path)
