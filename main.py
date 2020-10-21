import argparse
import datetime
import math
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from averaging import update_avg_net
from models import TwoLayerNN
from utils import get_data, accuracy


def eval(eval_loader, net, crit, args):

    net.eval()
    # run over both test and train set
    total_size = 0
    total_loss = 0
    total_acc = 0
    outputs = []

    with torch.no_grad():
        P = 0  # num samples / batch size
        for x, y in eval_loader:
            P += 1
            # loop over dataset
            x, y = x.to(args.device), y.to(args.device)
            out = net(x)
            outputs.append(out)

            loss = crit(out, y)
            prec = accuracy(out, y)
            bs = x.size(0)

            total_size += int(bs)
            total_loss += float(loss) * bs
            total_acc += float(prec) * bs

        hist = [total_loss / total_size, total_acc / total_size]
        print(hist)

        return hist, outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', default=100000, type=int)
    parser.add_argument('--batch_size_train', default=100, type=int)
    parser.add_argument('--batch_size_eval', default=100, type=int,
                        help='must be equal to training batch size')
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--mom', default=0, type=float)
    parser.add_argument('--wd', default=0, type=float)
    parser.add_argument('--print_freq', default=200, type=int)
    parser.add_argument('--eval_freq', default=200, type=int)
    parser.add_argument('--dataset', default='mnist', type=str,
                        help='mnist | cifar10 | cifar100')
    parser.add_argument('--path', default='./data', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', default='fc', type=str)
    parser.add_argument('--criterion', default='NLL', type=str,
                        help='NLL | linear_hinge')
    parser.add_argument('--width', default=100, type=int,
                        help='width of fully connected layers')
    parser.add_argument('--save_dir', default='results/', type=str)
    parser.add_argument('--custom_init', action='store_true', default=False)
    parser.add_argument('--traj', action='store_true', default=False)
    parser.add_argument('--pca', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--double', action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--schedule', action='store_true', default=False)
    parser.add_argument('--preprocess', action='store_true', default=False)
    parser.add_argument('--lr_schedule', action='store_true', default=False)

    args = parser.parse_args()

    begin_time = time.time()

    # initial setup
    if args.double:
        torch.set_default_tensor_type('torch.DoubleTensor')
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')

    def init_weights(m):
        if type(m) == nn.Linear:
            #nn.init.xavier_uniform_(m.weight)
            m.weight.data.fill_(0.01)
            #nn.init.uniform_(m.weight, a=-1e-6, b=1e-6)

    # torch.manual_seed(args.seed)
    print(args)

    # training setup
    train_loader, test_loader_eval, train_loader_eval, num_classes, input_dim = get_data(args)

    net = TwoLayerNN(input_dim=input_dim, width=args.width, num_classes=num_classes).to(args.device)

    if args.custom_init:
        net.apply(init_weights)

    file_traj = args.save_dir + '_traj.log'
    f = open(file_traj, 'w+')
    f.write(str(args))
    f.write(str(net))
    f.close()

    avg_net = net

    print(net)

    opt = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=args.mom,
        weight_decay=args.wd
    )

    if args.criterion == 'NLL':
        crit = nn.CrossEntropyLoss(reduction='mean').to(args.device)

    def cycle_loader(dataloader):
        while 1:
            for data in dataloader:
                yield data

    circ_train_loader = cycle_loader(train_loader)

    # training logs per iteration
    training_history = []
    weight_grad_history = []

    # eval logs less frequently
    evaluation_history_TEST = []
    evaluation_history_TRAIN = []
    evaluation_history_AVG = []
    evaluation_history_AVGTRAIN = []

    STOP = False

    for i, (x, y) in enumerate(circ_train_loader):

        if i % args.eval_freq == 0:
            # first record is at the initial point
            print('test')
            te_hist, te_outputs = eval(test_loader_eval, net, crit, args)
            print('train eval')
            tr_hist, tr_outputs = eval(train_loader_eval, net, crit, args)
            if args.traj:
                print('train eval avg_net')
                tat_hist, tat_outputs = eval(train_loader_eval, avg_net, crit, args)
                evaluation_history_AVGTRAIN.append([i, *tat_hist])
                print('test eval avg_net')
                ta_hist, ta_outputs = eval(test_loader_eval, avg_net, crit, args)
                evaluation_history_AVG.append([i, *ta_hist])

            evaluation_history_TEST.append([i, *te_hist])
            evaluation_history_TRAIN.append([i, *tr_hist])

            # use traj file
            if args.traj:
                f = open(file_traj, 'a+')
                f.write('## Iteration {:d} \n'.format(i))
                f.write('Training set:\n')
                f.write(str(tr_hist) + '\n')
                f.write('Test set:\n')
                f.write(str(te_hist) + '\n')
                f.write('Avg train set: \n')
                f.write(str(tat_hist) + '\n')
                f.write('Avg test set: \n')
                f.write(str(ta_hist) + '\n')
                f.write('lr: ' + str(opt.param_groups[0]['lr']) + '\n')
                f.write('\n')
                f.close()

        net.train()

        x, y = x.to(args.device), y.to(args.device)

        opt.zero_grad()
        out = net(x)
        loss = crit(out, y)

        # calculate the gradients
        loss.backward()

        # record training history (starts at initial point)
        training_history.append([i, loss.item(), accuracy(out, y).item()])

        if args.alpha > 0:
            for group in opt.param_groups:
                gan = (args.lr / args.width) ** (1 / (1 - args.alpha))
                group['lr'] = args.lr * (i + (1 / gan)) ** (- args.alpha)

        # compute mean of the network over the trajectories
        if args.traj:
            avg_net = update_avg_net(net, avg_net, i, 1000)

        # take the step
        opt.step()

        if i % args.print_freq == 0:
            print(training_history[-1])
            print("lr: ", opt.param_groups[0]['lr'])

        if args.lr_schedule:
            scheduler.step(i)

        if i > args.iterations:
            STOP = True

        if STOP:
            # final evaluation and saving results
            print('eval time {}'.format(i))
            print('test')
            te_hist, te_outputs = eval(test_loader_eval, net, crit, args)
            print('train eval')
            tr_hist, tr_outputs = eval(train_loader_eval, net, crit, args)

            if args.traj:
                print('train eval avg_net')
                tat_hist, tat_outputs = eval(train_loader_eval, avg_net, crit, args)
                evaluation_history_AVGTRAIN.append([i + 1, *tat_hist])
                print('test eval avg_net')
                ta_hist, ta_outputs = eval(test_loader_eval, avg_net, crit, args)
                evaluation_history_AVG.append([i + 1, *ta_hist])

            evaluation_history_TEST.append([i + 1, *te_hist])
            evaluation_history_TRAIN.append([i + 1, *tr_hist])

            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            else:
                print('Folder already exists, beware of overriding old data!')

            # save the setup
            torch.save(args, args.save_dir + '/args.info')
            # save the outputs
            torch.save(te_outputs, args.save_dir + '/te_outputs.pyT')
            torch.save(tr_outputs, args.save_dir + '/tr_outputs.pyT')

            if args.traj:
                torch.save(ta_outputs, args.save_dir + '/ta_outputs.pyT')
                torch.save(evaluation_history_AVG, args.save_dir + '/evaluation_history_AVG.hist')
                torch.save(tat_outputs, args.save_dir + '/tat_outputs.pyT')
                torch.save(evaluation_history_AVGTRAIN, args.save_dir + '/evaluation_history_AVGTRAIN.hist')
                torch.save(avg_net, args.save_dir + '/avg_net.pyT')

            # save the model
            torch.save(net, args.save_dir + '/net.pyT')
            # save the logs
            torch.save(training_history, args.save_dir + '/training_history.hist')
            torch.save(evaluation_history_TEST, args.save_dir + '/evaluation_history_TEST.hist')
            torch.save(evaluation_history_TRAIN, args.save_dir + '/evaluation_history_TRAIN.hist')

            # use traj file
            if args.traj:
                f = open(file_traj, 'a+')
                f.write('## End \n')
                f.write('Training set:\n')
                f.write(str(tr_hist) + '\n')
                f.write('Test set:\n')
                f.write(str(te_hist) + '\n')
                f.write('Avg train set: \n')
                f.write(str(tat_hist) + '\n')
                f.write('Avg test set: \n')
                f.write(str(ta_hist) + '\n')
                f.write('\n')
                f.close()

            break

    print("Final evaluation: ")
    te_hist, te_outputs = eval(test_loader_eval, net, crit, args)
    tr_hist, tr_outputs = eval(train_loader_eval, net, crit, args)
    if args.traj:
        ta_hist, ta_outputs = eval(test_loader_eval, avg_net, crit, args)
        tat_hist, tat_outputs = eval(train_loader_eval, avg_net, crit, args)

    end_time = time.time()
    total_time = end_time - begin_time
    print("Total Time: " + str(datetime.timedelta(seconds=total_time)))
