import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from nnue_dataset import NNUEGameDataset
from nnue_network import NNUE


def main(args):
    logname = f'NNUE_{args.t_tuple[0]}_{args.t_tuple[1]}' \
              f'bs{args.batch_size}lr{args.lr}d{args.decay}'
    print(logname)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))
    torch.backends.cudnn.benchmark = True

    m = NNUE()
    # m.load_state_dict(torch.load(f'../models/{args.weights}.pt', map_location=device))
    m.to(device)
    m.train()

    t_dataset = NNUEGameDataset(args.t_tuple[0], args.t_tuple[1], device=device, augment=True)
    v_dataset = NNUEGameDataset(args.v_tuple[0], args.v_tuple[1], device=device, augment=False)
    t_dataloader = DataLoader(t_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    v_dataloader = DataLoader(v_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(m.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.decay)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, 7, gamma=0.1)
    t_loss = []
    v_loss = []
    t_len = len(t_dataloader)
    v_len = len(v_dataloader)
    v_counter = 0
    for epoch in range(args.epochs):
        print('-' * 10)
        print(f'Epoch: {epoch+1}')
        for x, y in tqdm(t_dataloader):
            optimizer.zero_grad()
            z = m(x)
            loss = loss_fn(z, y)
            t_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            v_counter += 1
            if v_counter % 20 == 0:
                m.eval()
                running_loss = 0.0
                with torch.no_grad():
                    for x, y in v_dataloader:
                        z = m(x)
                        loss = loss_fn(z, y)
                        running_loss += loss.item()
                v_loss.append(running_loss / v_len)
                m.train()

        print('Train loss {:.3f}'.format(np.mean(t_loss[-t_len:])))
        print('Valid loss {:.3f}'.format(np.mean(v_loss[-t_len//20:])))
        sched.step()
        if not args.nosave:
            torch.save(m.state_dict(), f'../models/{logname}e{epoch+1}.pt')

    np.savez(f'../logs/{logname}e{args.epochs}',
             t_loss=t_loss,
             v_loss=v_loss,
             params=args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--v_tuple', type=int, nargs=2, default=(0, 100),
                   help='tuple for validation data range')
    p.add_argument('--t_tuple', type=int, nargs=2, default=(100, 1000),
                   help='tuple for training data range')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=2048)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--decay', type=float, default=1e-3)
    p.add_argument('--nosave', action='store_true')
    # p.add_argument('--weights', type=str, default='init_ab',
    #                help='Path to starting weights')

    main(p.parse_args())
