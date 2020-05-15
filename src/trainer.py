import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import GameDataset
from network import UTTTNet


def main(args):
    logname = f'{args.t_tuple[0]}_{args.t_tuple[1]}' \
              f'bs{args.batch_size}lr{args.lr}d{args.decay}e{args.epochs}'
    print(logname)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))
    torch.backends.cudnn.benchmark = True

    m = UTTTNet()
    m.load_state_dict(torch.load(f'../models/{args.weights}.pt', map_location=device))
    m.to(device)

    t_dataset = GameDataset(args.t_tuple[0], args.t_tuple[1], device, augment=True)
    v_dataset = GameDataset(args.v_tuple[0], args.v_tuple[1], device, augment=False)
    t_dataloader = DataLoader(t_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    v_dataloader = DataLoader(v_dataset, batch_size=args.batch_size//8, shuffle=True)

    value_loss = nn.MSELoss()
    optimizer = torch.optim.SGD(m.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.decay)

    t_loss = []
    v_loss = []
    t_len = len(t_dataloader)
    v_len = len(v_dataloader)
    for epoch in range(args.epochs):
        print('-' * 10)
        print(f'Epoch: {epoch}')
        for x, (y_p, y_v) in tqdm(t_dataloader):
            m.train()
            optimizer.zero_grad()
            p, v = m(x)
            loss = value_loss(v, y_v) - torch.mean(y_p * p)  # p is log prob, y_p is not
            t_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            m.eval()
            running_loss = 0.0
            with torch.no_grad():
                for x, (y_p, y_v) in v_dataloader:
                    p, v = m(x)
                    loss = value_loss(v, y_v) - torch.mean(y_p * p)  # p is log prob, y_p is not
                    running_loss += loss.item()
            v_loss.append(running_loss / v_len)

        print('Train loss {:.3f}'.format(np.mean(t_loss[-t_len:])))
        print('Valid loss {:.3f}'.format(np.mean(v_loss[-t_len:])))

        torch.save(m.state_dict(), f'../models/{logname}_e{epoch}.pt')
        
    np.savez('../logs/' + logname,
             t_loss=t_loss,
             v_loss=v_loss,
             params=args)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--v_tuple', type=int, nargs=2, default=(0, 20),
                   help='tuple for validation data range')
    p.add_argument('--t_tuple', type=int, nargs=2, default=(20, 200),
                   help='tuple for training data range')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=2048)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--decay', type=float, default=1e-3)
    p.add_argument('--weights', type=str, default='00000',
                   help='Path to starting weights')

    main(p.parse_args())
