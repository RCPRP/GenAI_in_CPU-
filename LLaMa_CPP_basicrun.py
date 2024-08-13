#Colab version https://colab.research.google.com/drive/1rC68wYj0T2pipqCMbAmOtPahwO0y6kQk#scrollTo=UXJb1FtXgKii
pip install llama-cpp-python==0.1.78
pip install huggingface_hub

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # the model is in bin format
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

from llama_cpp import Llama

lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2, # CPU cores
    )
prompt = "Write a short poem about the independence of India"
prompt_template=f'''SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully.

USER: {prompt}

ASSISTANT:
'''

#output : AVX = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 | 



response = lcpp_llm(
    prompt=prompt_template,
    max_tokens=256,
    temperature=0.5,
    top_p=0.95,
    repeat_penalty=1.2,
    top_k=50,
    stop = ['USER:'], # Dynamic stopping when such token is detected.
    echo=True # return the prompt
)

print(response["choices"][0]["text"])

''' Took 17 sec in colab CPU
 Llama.generate: prefix-match hit
SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully.

USER: Write a short poem about the independence of India

ASSISTANT:

Today we celebrate, my dear friend,
The independence of our nation's end.
A long struggle came to an end,
With freedom in our hands, we transcend.

From British rule, we break free,
And forge a path that's uniquely me.
With pride and joy, we stand tall,
As one united nation, we give our all.

So let us raise our flags high,
And celebrate this special day in the sky.
For independence is a gift from above,
And we cherish it with love and devotion.
'''
