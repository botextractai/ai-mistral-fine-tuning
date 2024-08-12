import json
import os
import time
from datasets import Dataset
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.evaluation import DatasetGenerator
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.finetuning.callbacks import MistralAIFineTuningHandler
from llama_index.finetuning.mistralai import MistralAIFinetuneEngine
from llama_index.llms.mistralai import MistralAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness
from tqdm import tqdm
from typing import List

os.environ["MISTRAL_API_KEY"] = "REPLACE_THIS_WITH_YOUR_MISTRAL_API_KEY"
os.environ["OPENAI_API_KEY"] = "REPLACE_THIS_WITH_YOUR_OPENAI_API_KEY"    # Used for evaluation using RAGAS
WANDB_API_KEY = "REPLACE_THIS_WITH_YOUR_WANDB_API_KEY"

# Load Data
# =========

documents = SimpleDirectoryReader(
    input_files=["IPCC_AR6_WGII_Chapter03.pdf"]
).load_data()

# Setup LLMs and embedding model
# ==============================

open_mistral = MistralAI(
    model="open-mistral-7b", temperature=0.1
)  # model to be fine-tuned
mistral_small = MistralAI(
    model="mistral-small-latest", temperature=0.1
)  # model for question generation
embed_model = MistralAIEmbedding()

# Training and evaluation data generation
# =======================================

question_gen_query = (
    "You are a Teacher/ Professor. Your task is to setup "
    "a quiz/examination. Using the provided context, formulate "
    "a single question that captures an important fact from the "
    "context. Restrict the question to the context information provided."
    "You should generate only question and nothing else."
)

dataset_generator = DatasetGenerator.from_documents(
    documents[:80],
    question_gen_query=question_gen_query,
    llm=mistral_small,
)

# this might take a while
questions = dataset_generator.generate_questions_from_nodes(num=40)
print("Generated ", len(questions), " training questions")

with open("train_questions.txt", "w", encoding="utf-8") as f:
    for question in questions:
        f.write(question + "\n")

dataset_generator = DatasetGenerator.from_documents(
    documents[80:],
    question_gen_query=question_gen_query,
    llm=mistral_small,
)

# this might take a while
questions = dataset_generator.generate_questions_from_nodes(num=40)
print("Generated ", len(questions), " evaluation questions")

with open("eval_questions.txt", "w", encoding="utf-8") as f:
    for question in questions:
        f.write(question + "\n")

# Initial evaluation with "open-mistral-7b" query engine
# ======================================================

questions = []
with open("eval_questions.txt", "r") as f:
    for line in f:
        questions.append(line.strip())

# limit the context window to 2048 tokens, so that refine is used
Settings.context_window = 2048
Settings.llm = open_mistral
Settings.embed_model = MistralAIEmbedding()

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine(similarity_top_k=2)

contexts = []
answers = []

for question in questions:
    response = query_engine.query(question)
    contexts.append([x.node.get_content() for x in response.source_nodes])
    answers.append(str(response))

ds = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
)

result = evaluate(ds, [answer_relevancy, faithfulness])

print("Score before fine-tuning (open-mistral-7b): " + json.dumps(result))

# Use "mistral-large-latest" to collect training data
# ===================================================

finetuning_handler = MistralAIFineTuningHandler()
callback_manager = CallbackManager([finetuning_handler])

llm = MistralAI(model="mistral-large-latest", temperature=0.1)
llm.callback_manager = callback_manager

questions = []
with open("train_questions.txt", "r") as f:
    for line in f:
        questions.append(line.strip())

Settings.embed_model = MistralAIEmbedding()
Settings.llm = llm

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine(similarity_top_k=2)

contexts = []
answers = []

for question in tqdm(questions, desc="Processing questions"):
    response = query_engine.query(question)
    contexts.append(
        "\n".join([x.node.get_content() for x in response.source_nodes])
    )
    answers.append(str(response))

def convert_data_jsonl_format(
    questions: List[str],
    contexts: List[str],
    answers: List[str],
    output_file: str,
) -> None:
    with open(output_file, "w") as outfile:
        for context, question, answer in zip(contexts, questions, answers):
            message_dict = {
                "messages": [
                    {
                        "role": "assistant",
                        "content": "You are a helpful assistant to answer user queries based on provided context.",
                    },
                    {
                        "role": "user",
                        "content": f"context: {context} \n\n question: {question}",
                    },
                    {"role": "assistant", "content": answer},
                ]
            }
            # write the JSON object in a single line
            json.dump(message_dict, outfile)
            # add a newline character after each JSON object
            outfile.write("\n")

convert_data_jsonl_format(questions, contexts, answers, "training.jsonl")

# Create "MistralAIFinetuneEngine"
# ================================

# Weights & Biases (WandB) for monitorning the training logs
wandb_integration_dict = {
    "project": "mistralai",
    "run_name": "fine-tuning",
    "api_key": WANDB_API_KEY,
}

finetuning_engine = MistralAIFinetuneEngine(
    base_model="open-mistral-7b",
    training_path="training.jsonl",
    # validation_path="<validation file>", # validation file is optional
    verbose=True,
    training_steps=5,
    learning_rate=0.0001,
    wandb_integration_dict=wandb_integration_dict,
)

# starts the fine-tuning of open-mistral-7b
finetuning_engine.finetune()

# this shows the current status of the job - 'RUNNING'
finetuning_engine.get_current_job()

# wait for job status - 'SUCCESS'
status = str(finetuning_engine.get_current_job())
while "SUCCESS" not in status:
    time.sleep(10)
    status = str(finetuning_engine.get_current_job())

ft_llm = finetuning_engine.get_finetuned_model(temperature=0.1)

# Evaluation
# ==========

# setting up fine-tuned LLM
Settings.llm = ft_llm
Settings.context_window = (
    2048  # limit the context window artifically to test refine process
)
Settings.embed_model = MistralAIEmbedding()

questions = []
with open("eval_questions.txt", "r") as f:
    for line in f:
        questions.append(line.strip())

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=2, llm=ft_llm)

contexts = []
answers = []

for question in tqdm(questions, desc="Processing Questions"):
    response = query_engine.query(question)
    contexts.append([x.node.get_content() for x in response.source_nodes])
    answers.append(str(response))

ds = Dataset.from_dict(
    {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
)

result = evaluate(ds, [answer_relevancy, faithfulness])

print("Score after fine-tuning (open-mistral-7b-finetuned): " + json.dumps(result))
