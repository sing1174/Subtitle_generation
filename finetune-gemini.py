# !pip -q install datasets transformers accelerate

import google.generativeai as genai
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments


# Configure the API key
api_key = "AIzaSyB2ApAg4Dk5ctVmhR0XHCKRv6dMGIrCtts"
genai.configure(api_key=api_key)


# Load the dataset
ds = load_dataset("cnbeining/sentence-segmentation")

# View dataset structure
print(ds)
print(ds["train"][0])  # View an example from the training set

# Preprocessing function to flatten the 'output' list
def preprocess(example):
    return {
        'input_text': example['input'],  # Keep input as-is
        'target_text': " ".join([" ".join(segment) for segment in example['output']])  # Flatten segments
    }

# Apply preprocessing
processed_dataset = ds.map(preprocess)

# Split dataset
train_data = processed_dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = train_data['train']
val_dataset = train_data['test']

# Initialize the model
model = genai.GenerativeModel('gemini-pro')
# Tokenizer 
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()