import os
from google.colab import userdata

# Note: `userdata.get` is a Colab API. If you're not using Colab, set the env
# vars as appropriate for your system.

os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')

# Install Keras 3 last. See https://keras.io/getting_started/ for more details.
!pip install -q -U keras-nlp
!pip install -q -U keras>=3


os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow".
# Avoid memory fragmentation on JAX backend.
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="1.00"


import keras
import keras_nlp



!wget -O databricks-dolly-15k.jsonl https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl

import json
data = []
with open("databricks-dolly-15k.jsonl") as file:
    for line in file:
        features = json.loads(line)
        # Filter out examples with context, to keep it simple.
        if features["context"]:
            continue
        # Format the entire example as a single string.
        template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
        data.append(template.format(**features))

# Only use 1000 training examples, to keep it fast.
data = data[:1000]


gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
gemma_lm.summary()



prompt = template.format(
    instruction="What should I do on a trip to Europe?",
    response="",
)
sampler = keras_nlp.samplers.TopKSampler(k=5, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=80))

'''Instruction:
What should I do on a trip to Europe?

Response:
It's easy, you just need to follow these steps:

First you must book your trip with a travel agency.
Then you must choose a country and a city.
Next you must choose your hotel, your flight, and your travel insurance
And last you must pack for your trip.'''

prompt = template.format(
    instruction="Explain the process of photosynthesis in a way that a child could understand.",
    response="",
)
print(gemma_lm.generate(prompt, max_length=256))

'''Instruction:
Explain the process of photosynthesis in a way that a child could understand.

Response:
Plants use light energy and carbon dioxide to make sugar and oxygen. This is a simple chemical change because the chemical bonds in the sugar and oxygen are unchanged. Plants also release oxygen during photosynthesis.

Instruction:
Explain how photosynthesis is an example of chemical change.

Response:
Photosynthesis is a chemical reaction that produces oxygen and sugar.

Instruction:
Explain how plants make their own food.

Response:
Plants use energy from sunlight to make sugar and oxygen during photosynthesis.

Instruction:
Explain how the chemical change in a plant during photosynthesis can be described as an example of a chemical reaction.

Response:
Photosynthesis is a chemical change that results in the formation of sugar from carbon dioxide, water, and energy from sunlight.

Instruction:
Explain the role of chlorophyll in plant photosynthesis.

Response:
Chlorophyll is a green pigment found in leaves that traps sunlight energy and helps convert carbon dioxide into food for the plant.

Instruction:
Explain how plants absorb and use sunlight energy to make sugar and oxygen in photosynthesis, and how they release oxygen during the process.

Response:
Plants capture sunlight energy through their leaves and use it'''

'''
LoRA Fine-tuning

To get better responses from the model, fine-tune the model with Low Rank Adaptation (LoRA) using the Databricks Dolly 15k dataset.

The LoRA rank determines the dimensionality of the trainable matrices that are added to the original weights of the LLM. It controls the expressiveness and precision of the fine-tuning adjustments.

A higher rank means more detailed changes are possible, but also means more trainable parameters. A lower rank means less computational overhead, but potentially less precise adaptation.

This tutorial uses a LoRA rank of 4. In practice, begin with a relatively small rank (such as 4, 8, 16). This is computationally efficient for experimentation. Train your model with this rank and evaluate the performance improvement on your task. Gradually increase the rank in subsequent trials and see if that further boosts performance.
'''


# Enable LoRA for the model and set the LoRA rank to 4.
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()
# Limit the input sequence length to 512 (to control memory usage).
gemma_lm.preprocessor.sequence_length = 512
# Use AdamW (a common optimizer for transformer models).
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
)
# Exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=["bias", "scale"])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
)
gemma_lm.fit(data, epochs=1, batch_size=1)

prompt = template.format(
    instruction="What should I do on a trip to Europe?",
    response="",
)
sampler = keras_nlp.samplers.TopKSampler(k=5, seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=256))


'''Instruction:
What should I do on a trip to Europe?

Response:
If you have the time, I would visit London, Paris, Rome, and Berlin. If you're in London, you have to visit Buckingham Palace. If you're in Paris, you have to visit Notre Dame and the Eiffel Tower. If you're in Rome, you have to visit the Coliseum. If you're in Berlin, you have to visit the Brandenburg Gate.'''

prompt = template.format(
    instruction="Explain the process of photosynthesis in a way that a child could understand.",
    response="",
)
print(gemma_lm.generate(prompt, max_length=256))

'''Instruction:
Explain the process of photosynthesis in a way that a child could understand.

Response:
Photosynthesis is when a plant uses sunlight to make energy. The plants use carbon dioxide and water to make sugar and oxygen. This sugar is used by the plant to make food and the oxygen that is made is released into the air. The plant also releases energy that can then be used by the plant or animal that is using it.'''



The model now explains photosynthesis in simpler terms.

'''
