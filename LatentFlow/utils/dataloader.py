from torch.utils.data import DataLoader 
from torch.utils.data import RandomSampler, SequentialSampler
def get_loaders(train_dataset, test_dataset, config):
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=RandomSampler(train_dataset),
        shuffle=False,
        pin_memory=False, 
        drop_last=False,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        sampler=RandomSampler(test_dataset),
        pin_memory=False,
    )
    return train_loader, val_loader