import torch
import torch.distributed as dist
from transformers import BertForSequenceClassification
from elixir.search import minimum_waste_search
from elixir.wrapper import ElixirModule, ElixirOptimizer

# Initialize distributed training
dist.init_process_group(backend='nccl')
world_size = dist.get_world_size()  # Total GPUs
world_group = dist.group.WORLD      # Default communication group

# Load model and optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-8)

# Elixir setup
sr = minimum_waste_search(model, world_size)
model = ElixirModule(model, sr, world_group)
optimizer = ElixirOptimizer(model, optimizer)
