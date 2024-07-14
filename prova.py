import torch

def lr_scheduler(optimizer: torch.optim.Optimizer,
                 initial_lr: float,
                 steady_lr: float,
                 final_lr: float,
                 total_epochs: int,
                 start_epoch: int = 0):
    
    increasing_epochs = int(total_epochs * 0.1)
    decreasing_epochs = total_epochs - increasing_epochs

    # Set initial learning rate of optimizer to 1
    for grp in optimizer.param_groups:
        grp["lr"] = 1.0

    # Define the schedulers with direct learning rates
    increasing_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=initial_lr,
        end_factor=steady_lr,
        total_iters=increasing_epochs,
    )

    decreasing_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=steady_lr,
        end_factor=final_lr,
        total_iters=decreasing_epochs,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[increasing_scheduler, decreasing_scheduler],
        milestones=[increasing_epochs],
        last_epoch=start_epoch - 1
    )

    return optimizer, scheduler

# Example usage:
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--initial_learning_rate', type=float, default=1e-4)
parser.add_argument('--steady_learning_rate', type=float, default=0.002)
parser.add_argument('--final_learning_rate', type=float, default=1e-6)
args = parser.parse_args()

INITIAL_LR = args.initial_learning_rate
STEADY_LR = args.steady_learning_rate
FINAL_LR = args.final_learning_rate
EPOCHS = 100  # Example value, you should set this according to your use case
TOTAL_FRACTIONS = 1  # Example value, you should set this according to your use case
FRACTION = 0  # Example value, you should set this according to your use case

model = torch.nn.Linear(10, 1)  # Example model
optimizer = torch.optim.Adam(params=model.parameters(), lr=1, weight_decay=0, betas=(0.9, 0.95))

optimizer, scheduler = lr_scheduler(
    optimizer=optimizer,
    initial_lr=INITIAL_LR,
    steady_lr=STEADY_LR,
    final_lr=FINAL_LR,
    total_epochs=EPOCHS * TOTAL_FRACTIONS,
    start_epoch=FRACTION * EPOCHS
)

# To verify the initial learning rate
print(scheduler.get_last_lr())  # This should print [0.0001]
scheduler.step()
print(scheduler.get_last_lr())  # This should print [0.0001]


