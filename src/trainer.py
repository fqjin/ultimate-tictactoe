import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import GameDataset
from network import UTTTNet


def main(args):
    logname = f'from{args.weights}_{args.t_tuple[0]}_{args.t_tuple[1]}' \
              f'bs{args.batch_size}lr{args.lr}d{args.decay}e{args.epochs}'
    print(logname)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))
    torch.backends.cudnn.benchmark = True

    m = UTTTNet()
    m.load_state_dict(torch.load(f'../models/{args.weights}.pt', map_location=device))
    m.to(device)

    dataset = GameDataset(args.t_tuple[0], args.t_tuple[1], device, augment=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    v_loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(m.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.decay)

    t_loss = []
    data_len = len(dataloader)
    for epoch in range(args.epochs):
        print('-' * 10)
        print(f'Epoch: {epoch}')

        m.train()
        for x, (y_p, y_v) in tqdm(dataloader):
            optimizer.zero_grad()
            p, v = m(x)
            loss = v_loss_fn(v, y_v) - torch.mean(y_p * p)  # p is log prob, y_p is not
            t_loss.append(loss.data.item())
            loss.backward()
            optimizer.step()
        print('Train loss {:.3f}'.format(np.mean(t_loss[-data_len:])))

        # No validation loop

    torch.save(m.state_dict(), f'../models/{logname}.pt')
    np.savez('../logs/' + logname, t_loss=t_loss, params=args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--t_tuple', type=int, nargs=2, default=(0, 100),
                   help='tuple for training data range')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=1024)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--decay', type=float, default=1e-5)
    # p.add_argument('--seed', type=int, default=0)
    p.add_argument('--weights', type=str, default='00000',
                   help='Path to starting weights')

    main(p.parse_args())
