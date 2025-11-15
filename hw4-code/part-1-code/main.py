import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from utils import *
import os

# Set seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Tokenize the input
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Core training function
def do_train(args, model, train_dataloader, save_dir="./out"):
    global device

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_epochs = args.num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    model.train()
    progress_bar = tqdm(range(num_training_steps))

    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Implement the training loop --- make sure to use the optimizer and lr_sceduler (learning rate scheduler)
    # Remember that pytorch uses gradient accumumlation so you need to use zero_grad (https://pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html)
    # You can use progress_bar.update(1) to see the progress during training
    # You can refer to the pytorch tutorial covered in class for reference

    cuda_failed = False
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                progress_bar.update(1)
            except RuntimeError as e:
                if "cudaErrorNoKernelImageForDevice" in str(e) and not cuda_failed:
                    print(f"CUDA error during training: {e}")
                    print("Switching to CPU and restarting training...")
                    cuda_failed = True
                    device = torch.device("cpu")
                    model = model.to(device)
                    # Reset optimizer and scheduler for CPU
                    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
                    lr_scheduler = get_scheduler(
                        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
                    )
                    progress_bar = tqdm(range(num_training_steps))
                    # Restart from beginning
                    return do_train(args, model, train_dataloader, save_dir)
                else:
                    raise e

    ##### YOUR CODE ENDS HERE ######

    print("Training completed...")
    print("Saving Model....")
    model.save_pretrained(save_dir)

    return


# Core evaluation function
def do_eval(eval_dataloader, output_dir, out_file):
    global device

    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    try:
        model.to(device)
    except RuntimeError as e:
        if "cudaErrorNoKernelImageForDevice" in str(e):
            print(f"CUDA error during eval model loading: {e}")
            print("Switching to CPU for evaluation...")
            device = torch.device("cpu")
            model.to(device)
        else:
            raise e

    model.eval()

    # Manually compute accuracy to avoid external metric loading issues
    correct = 0
    total = 0
    out_file = open(out_file, "w")

    for batch in tqdm(eval_dataloader):
        try:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
        except RuntimeError as e:
            if "cudaErrorNoKernelImageForDevice" in str(e):
                print(f"CUDA error during evaluation: {e}")
                print("Switching to CPU for evaluation...")
                device = torch.device("cpu")
                model = model.to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = model(**batch)
            else:
                raise e

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # Move to CPU and convert to plain Python ints to avoid any weird tensor issues
        preds_cpu = predictions.detach().cpu().tolist()
        labels_cpu = batch["labels"].detach().cpu().tolist()

        # Update manual accuracy counters
        for p, l in zip(preds_cpu, labels_cpu):
            if p == l:
                correct += 1
            total += 1

            # write to output file (pred then gold label)
            out_file.write(f"{int(p)}\n")
            out_file.write(f"{int(l)}\n")
    out_file.close()

    # Return accuracy in the same dict format as evaluate would
    score = {"accuracy": correct / total if total > 0 else 0.0}

    return score


# Created a dataladoer for the augmented training dataset
def create_augmented_dataloader(args, dataset):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Here, 'dataset' is the original dataset. You should return a dataloader called 'train_dataloader' -- this
    # dataloader will be for the original training split augmented with 5k random transformed examples from the training set.
    # You may find it helpful to see how the dataloader was created at other place in this code.

    # Get original training set
    original_train = dataset["train"]

    # Select 5k random examples and transform them
    augmented_examples = dataset["train"].shuffle(seed=42).select(range(5000))
    augmented_examples = augmented_examples.map(custom_transform)

    # Combine original and augmented datasets
    combined_dataset = datasets.concatenate_datasets([original_train, augmented_examples])

    # Tokenize the combined dataset
    combined_tokenized = combined_dataset.map(tokenize_function, batched=True)
    combined_tokenized = combined_tokenized.remove_columns(["text"])
    combined_tokenized = combined_tokenized.rename_column("label", "labels")
    combined_tokenized.set_format("torch")

    # Create dataloader
    train_dataloader = DataLoader(combined_tokenized, shuffle=True, batch_size=args.batch_size, num_workers=4)

    ##### YOUR CODE ENDS HERE ######

    return train_dataloader


# Create a dataloader for the transformed test set
def create_transformed_dataloader(args, dataset, debug_transformation):
    # Print 5 random transformed examples
    if debug_transformation:
        small_dataset = dataset["test"].shuffle(seed=42).select(range(5))
        small_transformed_dataset = small_dataset.map(custom_transform)
        for k in range(5):
            print("Original Example ", str(k))
            print(small_dataset[k])
            print("\n")
            print("Transformed Example ", str(k))
            print(small_transformed_dataset[k])
            print('=' * 30)

        exit()

    transformed_dataset = dataset["test"].map(custom_transform)
    transformed_tokenized_dataset = transformed_dataset.map(tokenize_function, batched=True)
    transformed_tokenized_dataset = transformed_tokenized_dataset.remove_columns(["text"])
    transformed_tokenized_dataset = transformed_tokenized_dataset.rename_column("label", "labels")
    transformed_tokenized_dataset.set_format("torch")

    transformed_val_dataset = transformed_tokenized_dataset
    eval_dataloader = DataLoader(transformed_val_dataset, batch_size=args.batch_size, num_workers=4)

    return eval_dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--train", action="store_true", help="train a model on the training data")
    parser.add_argument("--train_augmented", action="store_true", help="train a model on the augmented training data")
    parser.add_argument("--eval", action="store_true", help="evaluate model on the test set")
    parser.add_argument("--eval_transformed", action="store_true", help="evaluate model on the transformed test set")
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--debug_train", action="store_true",
                        help="use a subset for training to debug your training loop")
    parser.add_argument("--debug_transformation", action="store_true",
                        help="print a few transformed examples for debugging")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()

    global device
    global tokenizer

    # Device - Check for CUDA compatibility issues
    if torch.cuda.is_available():
        try:
            # Test CUDA compatibility more thoroughly
            test_tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
            test_tensor = test_tensor.to('cuda')
            # Perform an operation that requires CUDA kernels
            result = test_tensor.sum()
            result.backward()
            device = torch.device("cuda")
            print("Using CUDA device")
        except RuntimeError as e:
            print(f"CUDA not compatible: {e}")
            print("Falling back to CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # Load IMDB dataset from local files
    import os
    import glob
    from datasets import Dataset, DatasetDict

    def load_imdb_from_local(data_dir="data/aclImdb"):
        """Load IMDB dataset from local files instead of using datasets library"""
        train_texts = []
        train_labels = []
        test_texts = []
        test_labels = []

        # Load training data
        for label, label_name in enumerate(['neg', 'pos']):
            pattern = os.path.join(data_dir, 'train', label_name, '*.txt')
            for filepath in glob.glob(pattern):
                with open(filepath, 'r', encoding='utf-8') as f:
                    train_texts.append(f.read())
                    train_labels.append(label)

        # Load test data
        for label, label_name in enumerate(['neg', 'pos']):
            pattern = os.path.join(data_dir, 'test', label_name, '*.txt')
            for filepath in glob.glob(pattern):
                with open(filepath, 'r', encoding='utf-8') as f:
                    test_texts.append(f.read())
                    test_labels.append(label)

        train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
        test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})

        return DatasetDict({'train': train_dataset, 'test': test_dataset})

    dataset = load_imdb_from_local()
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Prepare dataset for use by model
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(4000))
    small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

    # Create dataloaders for iterating over the dataset
    if args.debug_train:
        train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4)
        eval_dataloader = DataLoader(small_eval_dataset, batch_size=args.batch_size, num_workers=4)
        print(f"Debug training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")
    else:
        train_dataloader = DataLoader(tokenized_dataset["train"], shuffle=True, batch_size=args.batch_size, num_workers=4)
        eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=args.batch_size, num_workers=4)
        print(f"Actual training...")
        print(f"len(train_dataloader): {len(train_dataloader)}")
        print(f"len(eval_dataloader): {len(eval_dataloader)}")

    # Train model on the original training dataset
    if args.train:
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        try:
            model.to(device)
        except RuntimeError as e:
            if "cudaErrorNoKernelImageForDevice" in str(e):
                print(f"CUDA error during model loading: {e}")
                print("Switching to CPU...")
                device = torch.device("cpu")
                model.to(device)
            else:
                raise e
        do_train(args, model, train_dataloader, save_dir="./out")
        # Change eval dir
        args.model_dir = "./out"

    # Train model on the augmented training dataset
    if args.train_augmented:
        train_dataloader = create_augmented_dataloader(args, dataset)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
        try:
            model.to(device)
        except RuntimeError as e:
            if "cudaErrorNoKernelImageForDevice" in str(e):
                print(f"CUDA error during model loading: {e}")
                print("Switching to CPU...")
                device = torch.device("cpu")
                model.to(device)
            else:
                raise e
        do_train(args, model, train_dataloader, save_dir="./out_augmented")
        # Change eval dir
        args.model_dir = "./out_augmented"

    # Evaluate the trained model on the original test dataset
    if args.eval:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + "_original.txt"
        score = do_eval(eval_dataloader, args.model_dir, out_file)
        print("Score: ", score)

    # Evaluate the trained model on the transformed test dataset
    if args.eval_transformed:
        out_file = os.path.basename(os.path.normpath(args.model_dir))
        out_file = out_file + "_transformed.txt"
        eval_transformed_dataloader = create_transformed_dataloader(args, dataset, args.debug_transformation)
        score = do_eval(eval_transformed_dataloader, args.model_dir, out_file)
        print("Score: ", score)
