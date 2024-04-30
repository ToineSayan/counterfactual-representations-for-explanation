from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("CEBaB/CEBaB")
dataset = dataset.rename_column('review_majority', 'labels')
dataset = dataset.filter(lambda example : example["labels"] != 'no majority')
# dataset = dataset.cast(Features({"a": Value("float32")}))
dataset = dataset.class_encode_column('labels')
print(dataset)

# train_set = 'train_exclusive'
train_set = 'train_inclusive'
# train_set = 'train_observational'
eval_set = 'validation'

train_data = dataset[train_set]
test_data = dataset[eval_set]

# Define the BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Tokenize data
def tokenize_data(examples):
    return tokenizer(examples['description'], padding='max_length', truncation=True, max_length=128)

train_data = train_data.map(tokenize_data, batched=True, batch_size=32)
test_data = test_data.map(tokenize_data, batched=True, batch_size=32)

train_data = train_data.remove_columns([l for l in dataset[train_set].column_names if not l == 'labels'])
test_data = test_data.remove_columns([l for l in dataset[eval_set].column_names if not l == 'labels'])


# Define training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    output_dir="./results",
    load_best_model_at_end=True,
    save_strategy="epoch"
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    tokenizer=tokenizer,
    compute_metrics=lambda pred: {"accuracy": (pred.label_ids == pred.predictions.argmax(-1)).mean()}
)

# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)

trainer.save_model(f'./finetuned_bert_uncased_cebab_{train_set}')


