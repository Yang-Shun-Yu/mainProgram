from argparse import ArgumentParser
import datasets
import Transforms
from model import make_model
from trainer import AICUPTrainer
from torch.optim import SGD,AdamW
import os
from torch.nn import CrossEntropyLoss, TripletMarginLoss

from CenterLoss import CenterLoss
import Scheduler
import torch
import numpy as np

if __name__ == '__main__':
    seed = 0xdc51ab
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../AICUP_datasets')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--batch_size', '-b', type=int, default=24)
    parser.add_argument('--lr', '-l', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '--wd', default=1e-6)
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--smoothing', type=float, default=0)
    parser.add_argument('--margin', '-m', type=float, default=0.6)
    parser.add_argument('--save_dir', '-s', type=str, required=True)
    parser.add_argument('--check_init', action='store_true')
    parser.add_argument('--backbone', type=str)

    parser.add_argument('--embedding_dim', type=int, default=2048)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--center_loss', action='store_true')
    parser.add_argument(
        '--similarity', 
        choices=['euclidean', 'cosine'], 
        default='cosine', 
        help="Choose the similarity metric to use: 'euclidean' or 'cosine'. Default is 'euclidean'."
    )
    parser.add_argument('--custom_weights', type=str, default=None, help='Path to custom model weights')


    args = parser.parse_args()

    train_dataloader,total_class= datasets.get_AICUP_train(
        AICUP_path=args.dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        transform=Transforms.get_training_transform(),
        shuffle=True
    )

    gallery_dataloader, query_dataloader = datasets.get_AICUP_valid(
        AICUP_path=args.dataset, 
        num_workers=args.workers, 
        batch_size=args.batch_size, 
        transform=Transforms.get_valid_transform()
    )

    net = make_model(backbone=args.backbone, num_classes=total_class)
    print(total_class)
    # print(net)
    if args.custom_weights:
        print(f"Loading custom weights from {args.custom_weights}")
        state_dict = torch.load(args.custom_weights, map_location='cpu')
        model_dict = net.state_dict()

        # Filter out the classification layer weights
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and 'classifier' not in k}

        # Update the existing model's state_dict
        model_dict.update(pretrained_dict)

        # Load the updated state_dict into the model
        net.load_state_dict(model_dict)

        print("Custom weights loaded successfully, excluding the classification layer.")
    # if args.center_loss:
    #     center_loss_fn=CenterLoss(num_classes=total_class, feat_dim=args.embedding_dim, use_gpu=True)
    #     optim = SGD([
    #     {'params': net.parameters()},
    #     {'params': center_loss_fn.parameters(), 'lr': 0.5}  # Set a separate learning rate for centers
    #     ], lr=args.lr, momentum=0.9)
    #     scheduler = Scheduler.get_scheduler_net_center(optim)

    # else:
    #     center_loss_fn = None
    #     optim = SGD(net.parameters(), lr=args.lr, momentum=0.9)
    #     scheduler = Scheduler.get_scheduler_net(optim)

    optim = AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.center_loss:
        center_loss_fn=CenterLoss(num_classes=total_class, feat_dim=args.embedding_dim, use_gpu=True)
        optimizer_center = torch.optim.SGD(center_loss_fn.parameters(), lr=0.5)
    else:
        center_loss_fn = None
        optimizer_center = None

    scheduler = None


    trainer = AICUPTrainer(
        backbone=args.backbone,
        net=net,
        ce_loss_fn=CrossEntropyLoss(label_smoothing=args.smoothing),
        triplet_loss_fn=TripletMarginLoss(margin=args.margin),
        center_loss_fn=center_loss_fn, 
        optimizer=optim,
        optimizer_center=optimizer_center,
        lr_scheduler=scheduler,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    trainer.fit(
        train_dataloader=train_dataloader,
        gallery_dataloader=gallery_dataloader,
        query_dataloader=query_dataloader,
        epochs=args.epochs,
        early_stopping=args.early_stopping,
        save_dir=args.save_dir,
        check_init=args.check_init,
        similarity=args.similarity
        
    )
  






