import sys
sys.path.append("../")

import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from erwin.models.erwin import ErwinTransformer
from erwin.models.transolver import Transolver
from erwin.experiments.training import fit, create_scheduler_with_warmup
from erwin.experiments.datasets.elas import ElasDataset
from erwin.experiments.wrappers import ElasModel

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="erwin", 
                        choices=('mpnn', 'pointtransformer', 'pointnetpp', 'erwin', 'transolver'))
    parser.add_argument("--data-path", type=str, default='/home/mzhdano/data/standard_pde')
    parser.add_argument("--size", type=str, default="small", 
                        choices=('small', 'medium', 'large'))
    parser.add_argument("--num-epochs", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--use-wandb", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--val-every-iter", type=int, default=1000, 
                        help="Validation frequency")
    parser.add_argument("--experiment", type=str, default="elas", 
                        help="Experiment name in wandb")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
                        help="Ratio of total steps to use for warmup if warmup-steps not specified")
    
    return parser.parse_args()


erwin_configs = {
    "small": {
        "c_in": 64,
        "c_hidden": [64, 64],
        "ball_sizes": [256, 256],
        "enc_num_heads": [8, 8],
        "enc_depths": [6, 6],
        "dec_num_heads": [8],
        "dec_depths": [6],
        "strides": [1],
        "rotate": 45,
        "mp_steps": 3,
        "dimensionality": 2,
    },
}

transolver_configs = {
    'small': {
        "c_in": 128, 
        "c_hidden": 128,
        "slice_num": 64,
        "num_heads": 8,
        "num_layers": 8,
        "dimensionality": 2,
        "mlp_ratio": 1,
    }
}

model_cls = {
    "erwin": ErwinTransformer,
    'transolver': Transolver,
}


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    train_dataset = ElasDataset(
        data_path=args.data_path,
        split="train",
        knn=args.knn,
    )

    valid_dataset = ElasDataset(
        data_path=args.data_path,
        split="test",
        knn=args.knn,
    )

    test_dataset = ElasDataset(
        data_path=args.data_path,
        split="test",
        knn=args.knn,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.batch_size,
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.batch_size,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.batch_size,
    )

    if args.model == "erwin":
        model_config = erwin_configs[args.size]
    elif args.model == "transolver":
        model_config = transolver_configs[args.size]
    else:
        raise NotImplementedError(f"Unknown model: {args.model}")
    
    main_model = model_cls[args.model](**model_config)
    model = ElasModel(main_model).cuda()
    model = torch.compile(model)

    total_steps = args.num_epochs * len(train_dataset) // args.batch_size
    args.num_epochs = total_steps

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    warmup_steps = int(total_steps * args.warmup_ratio) if args.warmup_ratio else 0
    scheduler = create_scheduler_with_warmup(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=1e-7
    )

    config = vars(args)
    config.update(model_config)
    config['warmup_steps'] = warmup_steps

    fit(config, model, optimizer, scheduler, train_loader, valid_loader, test_loader, 110, 160)