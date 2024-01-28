import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, DataCollatorWithPadding, DataCollatorForLanguageModeling
from datasets import load_dataset
from tqdm.auto import tqdm, trange

from smollama import Llama, LLaMAConfig, generate

import wandb

from typarse import BaseParser


class Parser(BaseParser):
    device: str = "cpu"
    batch_size: int = 32
    lr: float = 5e-5

    _abbrev = {
        "device": "d",
        "batch_size": "b",
        "lr": "l",
    }

if __name__ == "__main__":

    args = Parser()

    wandb.init(project="smollm-pretraining")

    wandb.config.device = args.device
    wandb.config.batch_size = args.batch_size
    wandb.config.lr = args.lr

    device = args.device

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


    dataset = load_dataset("roneneldan/TinyStories")


    def tokenize_function(examples):
        return tokenizer(examples["text"], add_special_tokens=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    tokenized_datasets.set_format("torch", columns=["input_ids"], device=device)

    # data_collator = DataCollatorWithPadding(tokenizer)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        return_tensors="pt",
        mlm=False
    )

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )

    # eval_dataloader = DataLoader(
    #     tokenized_datasets["validation"],
    #     batch_size=32,
    #     collate_fn=data_collator
    # )

    config = LLaMAConfig(
        block_size=2048,
        vocab_size=tokenizer.vocab_size + 1,
        n_layer=8,
        n_head=8,
        n_embd=128,
    )

    model = Llama(config)

    model = model.to(device)

    print("Before training: ")
    print(generate(model, tokenizer, 100, "Once upon a time", device=device))

    count = sum([p.numel() for p in model.parameters()])
    print(f"Number of parameters: {count / 1e6:.1f}M")

    loss_fct = CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=args.lr)

    batch = next(iter(train_dataloader))

    # Training loop
    # for i, batch in enumerate(pbar := tqdm(train_dataloader)):
    table = wandb.Table(columns=["generated_text"])

    for i in (pbar := trange(1000)):

        inputs = batch["input_ids"][:-1].to(device)
        labels = batch["labels"][1:].to(device)

        logits = model(inputs)
        loss = loss_fct(logits.view(-1, tokenizer.vocab_size + 64), labels.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()

        log = {"loss": loss_value}
        wandb.log(log)

        if i % 50 == 0:
            sample = generate(model, tokenizer, 100, "Once upon a time", device=device, disable_tqdm=True)
            table.add_data(sample)


        pbar.set_description(f"Loss: {loss_value:.4f}")

    final_sample = generate(model, tokenizer, 100, "Once upon a time", device=device, disable_tqdm=False)

    wandb.log({"generated_text": table})

    print("After training: ")
    print(final_sample)

    wandb.finish()


