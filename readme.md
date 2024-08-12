# Fine-tuning MistralAI Large Language Models (LLMs) with LlamaIndex and Weights & Biases (wandb.ai)

This LlamaIndex example shows how to fine-tune the `open-mistral-7b` LLM by using the MistralAI fine-tuning API.

The PDF document `IPCC_AR6_WGII_Chapter03.pdf` is used as the data source for the fine-tuning.

The `mistral-small-latest` LLM is used to generate training data. This example generates 40 training and 40 evaluation questions, both on different sections of the PDF document.

The untuned (original) `open-mistral-7b` LLM is then evaluated as the first baseline performance benchmark.

The `mistral-large-latest` LLM is then used to create synthetic training and evaluation questions to avoid any biases with the other LLMs. It generates the output file `training.jsonl`. This file is then used for fine-tuning the `open-mistral-7b` LLM.

The fine-tuning is done by using the `MistralAIFinetuneEngine` wrapper abstraction.

The fine-tuned new `ft:open-mistral-7b:xxxxxxxx:xxxxxxxx:xxxxxxxx` LLM is then evaluated, so that the results can be compared to the first benchmark of the untuned (original) LLM. This shows if the fine-tuned new LLM improved (or worsened) the accuracy and faithfulness compared to the untuned (original) LLM.

Evaluation is done using the [Ragas evaluation library](https://docs.ragas.io/en/stable/) using OpenAI.

The metrics can be monitored on Weights & Biases.

You need a Mistral API key for this project. [Get your Mistral API key here](https://auth.mistral.ai/ui/registration). You can insert your Mistral API key in the `main.py` script, or you can supply your Mistral AI key either via the `.env` file, or through an environment variable called `MISTRAL_API_KEY`.

You need also need OpenAI API key for this example. [Get your OpenAI API key here](https://platform.openai.com/login). You can insert your OpenAI API key in the `main.py` script, or you can supply your OpenAI API key either via the `.env` file, or through an environment variable called `OPENAI_API_KEY`.

Weights & Biases (W&B) is a MLOps platform that can help developers monitor and document Machine Learning training workflows from end to end. W&B is used to get an idea of how well the training is working and if the model is improving over time. You need a W&B API key for this project. [Get your free Weights & Biases API key here](https://wandb.ai/authorize). Insert your W&B API key in the `main.py` script.

You can check your W&B projects at https://wandb.ai/YOUR_WANDB_USER_NAME/projects .

## The results

This example measures the two key indicators "Relevancy" and "Faithfulness" before and after the LLM fine-tuning.

"Relevancy" measures how relevant the generated answer is to the prompt. If the generated answer is incomplete or contains redundant information, the score will be low. This is quantified by working out the chance of an LLM generating the given question using the generated answer. The answer is scaled to the (0,1) range. The higher the value, the better.

"Faithfulness" measures the factual consistency of the generated answer against the given context. This is done using a multi-step paradigm that includes creation of statements from the generated answer followed by verifying each of these statements against the context. The answer is scaled to the (0,1) range. The higher the value, the better.

```
Score before fine-tuning (open-mistral-7b): {"answer_relevancy": 0.8247644540593632, "faithfulness": 0.9296602387511479}
Score after fine-tuning (open-mistral-7b-finetuned): {"answer_relevancy": 0.8442738404243547, "faithfulness": 0.9635416666666667}
```

## Known issue

The `llama-index-finetuning` Python package currently contains this error: https://github.com/run-llama/llama_index/issues/14775 . This can be fixed by first installing the package and then editing the locally cached version:  
`.../llama-index-finetuning/llama_index/finetuning/mistralai/utils.py`

## Delete the fine-tuned LLM

The fine-tuned LLM can be deleted with this Python program code:  
`client.delete_model(retrieved_job.fine_tuned_model)`

Alternatively, the fine-tuned LLM can also deleted with a `DELETE` type API call to the endpoint of the fine-tuned LLM, for example:  
`https://api.mistral.ai/v1/models/ft:open-mistral-7b:xxxxxxxx:xxxxxxxx:xxxxxxxx`  
The API call must include thease two Headers (Key/Value pairs):  
`Accept: application/json`  
`Authorization: Bearer <YOUR_MISTRAL_API_KEY>`
