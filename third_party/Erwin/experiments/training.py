import wandb
import torch
import time
import os
from tqdm import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def create_scheduler_with_warmup(optimizer, warmup_steps, total_steps, eta_min=1e-7):
    """
    A scheduler with linear warmup followed by cosine annealing.
    
    Args:
        optimizer: The optimizer to schedule
        warmup_steps: Number of steps for linear warmup
        total_steps: Total number of training steps
        eta_min: Minimum learning rate for cosine annealing
    
    Returns:
        SequentialLR scheduler combining warmup and cosine annealing
    """
    # Linear warmup from 0 to base lr
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Cosine annealing for the remaining steps
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=eta_min
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    return scheduler


def setup_wandb_logging(model, config, project_name="ballformer"):
    wandb.init(project=project_name, config=config, name=config["model"] + '_' + config["experiment"])
    wandb.watch(model)
    wandb.config.update({"num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}, allow_val_change=True)


def get_checkpoint_path(config):
    """Get checkpoint path based on config"""
    save_dir = config.get('checkpoint_dir', 'checkpoints')
    if config['model'] in ['erwin', 'pointtransformer']:
        return os.path.join(save_dir, f"{config['model']}_{config['experiment']}_{config['size']}_{config['seed']}_best.pt")
    else:
        return os.path.join(save_dir, f"{config['model']}_{config['experiment']}_{config['seed']}_best.pt")


def save_checkpoint(model, optimizer, scheduler, config, val_loss, global_step):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'val_loss': val_loss,
        'global_step': global_step,
        'config': config
    }
    
    checkpoint_path = get_checkpoint_path(config)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    
    if config.get("use_wandb", False):
        wandb.log({"checkpoint/best_val_loss": val_loss}, step=global_step)


def load_checkpoint(model, optimizer, scheduler, config):
    checkpoint_path = get_checkpoint_path(config)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['val_loss'], checkpoint['global_step']


def train_step(model, batch, optimizer, scheduler, max_grad_norm):
    model.train()
    optimizer.zero_grad()
    
    batch_size = batch['batch_idx'][-1].item() + 1 if model.loss_reduce == "sum" else 1

    stat_dict = model.training_step(batch)
    (stat_dict["train/loss"] / batch_size).backward()

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    stat_dict['train/lr'] = optimizer.param_groups[0]['lr']
    return stat_dict


def validate(model, val_loader, config):
    model.eval()
    val_stats = {}
    num_batches = 0
    total_samples = 0
    
    use_tqdm = not config.get("use_wandb", False)
    iterator = tqdm(val_loader, desc="Validation") if use_tqdm else val_loader
    
    for batch in iterator:
        batch = {k: v.cuda() for k, v in batch.items()}
        stat_dict = model.validation_step(batch)
        
        current_batch_size = batch['batch_idx'][-1].item() + 1
        
        for k, v in stat_dict.items():
            if k not in val_stats:
                val_stats[k] = 0
            
            # For sum reduction, accumulate as-is (they're already sums)
            # For mean reduction, multiply by batch size to get sum, then accumulate
            if model.loss_reduce == "sum":
                val_stats[k] += v.cpu().detach()
            else:
                val_stats[k] += v.cpu().detach() * current_batch_size
        
        total_samples += current_batch_size
        
        if use_tqdm:
            iterator.set_postfix({"Loss": f"{stat_dict['val/loss'].item():.4f}"})
            
        num_batches += 1
    
    avg_stats = {f"avg/{k}": v / total_samples for k, v in val_stats.items()}
    return avg_stats


def fit(config, model, optimizer, scheduler, train_loader, val_loader, test_loader=None, timing_window_start=100, timing_window_size=500):
    if config.get("use_wandb", False):
        setup_wandb_logging(model, config)
    
    use_tqdm = not config.get("use_wandb", False)
    running_train_stats = {}
    samples_processed = 0
    global_step = 0
    best_val_loss = float('inf')
    max_steps = config["num_epochs"]
    
    while global_step < max_steps:
        iterator = tqdm(train_loader, desc=f"Training (step {global_step + 1}/{max_steps})") if use_tqdm else train_loader
        for batch in iterator:
            if global_step >= max_steps:
                break
                
            model.train()
            batch = {k: v.cuda() for k, v in batch.items()}

            # measure runtime statistics
            if global_step == timing_window_start:
                timing_start = time.perf_counter()
            
            if global_step == timing_window_start + timing_window_size:
                timing_end = time.perf_counter()
                total_time = timing_end - timing_start
                steps_per_second = timing_window_size / total_time
                if config.get("use_wandb", False):
                    wandb.log({"stats/steps_per_second": steps_per_second}, step=global_step)
                else:
                    print(f"Steps per second: {steps_per_second:.2f}")
            
            stat_dict = train_step(model, batch, optimizer, scheduler, config['max_grad_norm'])

            batch_size = batch['batch_idx'][-1].item() + 1
            samples_processed += batch_size
            
            for k, v in stat_dict.items():
                if "lr" not in k:
                    if k not in running_train_stats:
                        running_train_stats[k] = 0
                    
                    # Handle accumulation like in validation
                    if model.loss_reduce == "sum":
                        running_train_stats[k] += v.cpu().detach()
                    else:
                        running_train_stats[k] += v.cpu().detach() * batch_size
            
            if use_tqdm:
                loss_keys = [k for k in stat_dict.keys() if "loss" in k]
                iterator.set_postfix({
                    "step": f"{global_step + 1}/{max_steps}",
                    **{k: f"{stat_dict[k].item():.4f}" for k in loss_keys}
                })
            else:
                wandb.log({f"{k}": v.item() for k, v in stat_dict.items() if "lr" not in k}, step=global_step)
            
            # Validation and checkpointing
            if (global_step + 1) % config["val_every_iter"] == 0:
                train_stats = {f"avg/{k}": v / samples_processed for k, v in running_train_stats.items()}
                
                running_train_stats = {}
                samples_processed = 0
                
                val_stats = validate(model, val_loader, config)
                current_val_loss = val_stats['avg/val/loss']
                
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    save_checkpoint(model, optimizer, scheduler, config, best_val_loss, global_step)
                    if not config.get("use_wandb", False):
                        print(f"New best validation loss: {best_val_loss:.4f}, saved checkpoint")
                
                if config.get("use_wandb", False):
                    wandb.log({**train_stats, **val_stats, 'global_step': global_step}, step=global_step)
                else:
                    loss_keys = [k for k in val_stats.keys() if "loss" in k]
                    for k in loss_keys: 
                        print(f"Validation {k}: {val_stats[k]:.4f}")
            
            global_step += 1

    if test_loader is not None and config.get('test', False):
        print("Loading best checkpoint for testing...")
        best_val_loss, best_step = load_checkpoint(model, optimizer, scheduler, config)
        print(f"Loaded checkpoint from step {best_step} with validation loss {best_val_loss:.4f}")
        
        test_stats = validate(model, test_loader, config)
        if config.get("use_wandb", False):
            wandb.log({
                **{f"test/{k.replace('val/', '')}": v for k, v in test_stats.items()},
                'global_step': global_step
            }, step=global_step)
        else:
            loss_keys = [k for k in test_stats.keys() if "loss" in k]
            for k in loss_keys:
                print(f"Test {k}: {test_stats[k]:.4f}")
    return model