from argparse import ArgumentParser
import veri776
import Transforms
from model import make_model
from Trainer import ReIDTrainer
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
    parser.add_argument('--dataset', type=str, default='../veri776')
    parser.add_argument('--workers', type=int, default=20)
    parser.add_argument('--batch_size', '-b', type=int, default=24) # -b=20 is the limit on my machine (12GB GPU memory)
    parser.add_argument('--lr', '-l', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '--wd', default=1e-6)
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--smoothing', type=float, default=0)
    parser.add_argument('--margin', '-m', type=float, default=0.6)
    parser.add_argument('--save_dir', '-s', type=str, required=True)
    parser.add_argument('--check_init', action='store_true')
    # parser.add_argument('--backbone', type=str, choices=['resnet', 'resnext', 'seresnet', 'densenet', 'resnet34','swin','yolo','yolo11','resnet_a','resnet_b'], required=True)
    parser.add_argument('--backbone', type=str)

    parser.add_argument('--embedding_dim', type=int, default=2048)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--center_loss', action='store_true')
    # Add argument to choose the similarity metric
    parser.add_argument(
        '--similarity', 
        choices=['euclidean', 'cosine'], 
        default='cosine', 
        help="Choose the similarity metric to use: 'euclidean' or 'cosine'. Default is 'euclidean'."
    )


    args = parser.parse_args()


    # datasets
    train_loader = veri776.get_veri776_train(
        veri776_path=args.dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        transform=Transforms.get_training_transform(),
        shuffle=True
    )

    test_loader = veri776.get_veri776_test(
        veri_776_path=args.dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        transform=Transforms.get_test_transform()
    )
    # Create the DataLoader (you can adjust batch_size and num_workers as needed)
  

    net = make_model(backbone=args.backbone, num_classes=576)
    # print(net)
    # print(args.batch_size)



    # # Trainer
    # optim = SGD(net.parameters(), lr=args.lr)

    # if args.center_loss:
    #     center_loss_fn=CenterLoss(num_classes=576, feat_dim=args.embedding_dim, use_gpu=True)
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
    # optim = SGD(net.parameters(), lr=args.lr, momentum=0.9)
    scheduler = None
    # scheduler = Scheduler.get_scheduler_net(optim)
    if args.center_loss:
        center_loss_fn=CenterLoss(num_classes=576, feat_dim=args.embedding_dim, use_gpu=True)
        optimizer_center = torch.optim.SGD(center_loss_fn.parameters(), lr=0.5)


    else:
        center_loss_fn = None
        optimizer_center = None

    # AdamW optim
    # Trainer
    # if args.center_loss:
    #     center_loss_fn = CenterLoss(num_classes=576, feat_dim=args.embedding_dim, use_gpu=True)

    #     # Use AdamW optimizer with different parameter groups
    #     optim = AdamW([
    #         {'params': net.parameters(), 'lr': args.lr},  # Learning rate for main network parameters
    #         {'params': center_loss_fn.parameters(), 'lr': 0.5}  # Set a separate learning rate for centers
    #     ], lr=args.lr, weight_decay=args.weight_decay)  # You can adjust weight_decay to suit your needs

    #     # Assuming Scheduler has a function that works well with AdamW and the center loss scenario
    #     scheduler = Scheduler.get_scheduler_net_center(optim)

    # else:
    #     center_loss_fn = None
    #     # AdamW without center loss parameters
    #     optim = AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    #     # Use a scheduler designed for this setup
    #     scheduler = Scheduler.get_scheduler_net(optim)


    trainer = ReIDTrainer(
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
        train_loader=train_loader,
        test_loader=test_loader,
        gt_index_path=os.path.join(args.dataset, 'gt_index.txt'),
        name_query_path=os.path.join(args.dataset, 'name_query.txt'),
        jk_index_path=os.path.join(args.dataset, 'jk_index.txt'),
        epochs=args.epochs,
        early_stopping=args.early_stopping,
        save_dir=args.save_dir,
        check_init=args.check_init,
        similarity=args.similarity

    )
