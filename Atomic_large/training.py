import time
import math
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from Atomic-large import Atomic
from module import ModelConfig
from dataset import tokenizer, final_dataset
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

class Muon(torch.optim.Optimizer):
    @torch._dynamo.disable
    def __init__(self, params, lr=0.02, momentum=0.95, n_steps=5):
        """Профессиональный оптимизатор Muon для скрытых 2D-матриц весов"""
        defaults = dict(lr=lr, momentum=momentum, n_steps=n_steps)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            n_steps = group['n_steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                X = buf.view(buf.size(0), -1)
                X = X / (X.norm(p=2) + 1e-7)
                for _ in range(n_steps):
                    X = 1.5 * X - 0.5 * X @ X.t() @X
                p.data.add_(X.view_as(p.data), alpha=-lr)
MAX_STEPS = 5000
WARMUP_STEPS = 500
BATCH_SIZE = 16
EVAL_INTERVAL = 200
CHECKPOINT_DIR ='./Atomic'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
config = ModelConfig()
model = Atomic(config).to("cuda")
muon_params = []
adamw_params = []
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if param.ndim >= 2 and "embed" not in name and "head" not in name:
        muon_params.append(param)
    else:
        adamw_params.append(param)
MAX_LR_MUON = 0.015
MAX_LR_ADAMW = 2e-4
optimizer_muon = Muon(muon_params, lr=0.02)
optimizer_adamw = torch.optim.AdamW(adamw_params, lr=3e-4, weight_decay=0.01)
def get_lr(step, warmup_steps, max_steps, max_lr, min_lr_ratio=0.1):
    min_lr = max_lr * min_lr_ratio
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
final_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_loader = DataLoader(final_dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn = data_collator)
val_loader = DataLoader(final_dataset["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=data_collator)
def get_betch_generator(loader):
    while True:
        for batch in loader:
            yield batch["input_ids"].to("cuda"), batch["labels"].to("cuda")
train_batch_gen = get_betch_generator(train_loader)
@torch.no_grad()
def estimate_loss(model, loader, n_batches=10):
    model.eval()
    losses = []
    val_gen = get_betch_generator(loader)
    for _ in range(n_batches):
        x, y = next(val_gen)
        _, loss = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return np.mean(losses)

print("=== Start the training of hybrid model ===")
print(f"{'Step':>6} {'Muon LR':>9} {'AdamW LR':>9} {'Train Loss':>12} {'Val Loss':>10} {'Perplexity':>12}")
best_val_loss = float('inf')
start_time = time.time()
for step in range(MAX_STEPS):
    model.train()
    lr_muon = get_lr(step, WARMUP_STEPS,  MAX_STEPS, max_lr=MAX_LR_MUON)
    lr_adamw = get_lr(step, WARMUP_STEPS, MAX_STEPS, max_lr=MAX_LR_ADAMW)
    for param_group in optimizer_muon.param_groups:
        param_group['lr'] = lr_muon
    for param_group in optimizer_adamw.param_groups:
        param_group['lr'] = lr_adamw
    x, y = next(train_batch_gen)
    logits, loss = model(x, targets=y)
    optimizer_muon.zero_grad()
    optimizer_adamw.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer_muon.step()
    optimizer_adamw.step()
    if step % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
        val_loss = estimate_loss(model, val_loader)
        perplexity = math.exp(min(val_loss, 20))
        print(f"{step:>6} {lr_muon:>9.5f} {lr_adamw:>9.5f} {loss.item():>12.4f} {val_loss:>10.4f} {perplexity:>12.2f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_muon_state_dict': optimizer_muon.state_dict(),
                'optimizer_adamw_state_dict': optimizer_adamw.state_dict(),
                'val_loss': val_loss,
                'config': config,
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
elapsed = time.time() - start_time
print(f"\n🎉 Обучение полностью завершено за {elapsed:.1f}s")
print(f"🏆 Лучший Validation Loss модели: {best_val_loss:.4f}")
