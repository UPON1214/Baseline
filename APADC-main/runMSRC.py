import argparse
import collections
import itertools
import torch
import random

from modelhandwritten import Apadc
from get_indicator_matrix_A import get_mask
from util import cal_std, get_logger
from datasets import *
from configure import get_default_config


def main(MR=[0.3]):
    # Environments
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.devices)
    use_cuda = torch.cuda.is_available()
    print("GPU: " + str(use_cuda))
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    # Configure
    config = get_default_config(dataset)
    config['dataset'] = dataset
    print("Data set: " + config['dataset'])
    config['print_num'] = config['training']['epoch'] / 10     # print_num
    logger = get_logger()

    # Load data
    seed = config['training']['seed']
    X_list, Y_list = load_data(config)
    x1 = X_list[0].reshape(210, 24)
    x2 = X_list[1].reshape(210, 576)
    x3 = X_list[2].reshape(210, 512)
    x4 = X_list[3].reshape(210, 256)
    x5 = X_list[4].reshape(210, 254)
    x1_train_raw = torch.from_numpy(x1)
    x2_train_raw = torch.from_numpy(x2)
    x3_train_raw = torch.from_numpy(x3)
    x4_train_raw = torch.from_numpy(x4)
    x5_train_raw = torch.from_numpy(x5)

    for missingrate in MR:
        accumulated_metrics = collections.defaultdict(list)
        config['training']['missing_rate'] = missingrate
        print('--------------------Missing rate = ' + str(missingrate) + '--------------------')
        for data_seed in range(1, args.test_time + 1):
            # get the mask
            np.random.seed(data_seed)
            mask = get_mask(5, x1_train_raw.shape[0], config['training']['missing_rate'])
            # mask the data
            #x1_train = x1_train_raw * mask[:, 0][:, np.newaxis]
            #x2_train = x2_train_raw * mask[:, 1][:, np.newaxis]

            x1_train = torch.from_numpy((x1_train_raw * mask[:, 0][:, np.newaxis]).numpy().astype('float32')).to(device)
            x2_train = torch.from_numpy((x2_train_raw * mask[:, 1][:, np.newaxis]).numpy().astype('float32')).to(device)
            x3_train = torch.from_numpy((x3_train_raw * mask[:, 2][:, np.newaxis]).numpy().astype('float32')).to(device)
            x4_train = torch.from_numpy((x4_train_raw * mask[:, 3][:, np.newaxis]).numpy().astype('float32')).to(device)
            x5_train = torch.from_numpy((x5_train_raw * mask[:, 4][:, np.newaxis]).numpy().astype('float32')).to(device)

            #x1_train = torch.from_numpy(x1_train).float().to(device)
            #x2_train = torch.from_numpy(x2_train).float().to(device)
            mask = torch.from_numpy(mask).long().to(device)  # indicator matrix A

            # Set random seeds for model initialization
            np.random.seed(seed)
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True

            # Build the model
            APADC = Apadc(config)
            optimizer = torch.optim.Adam(
                itertools.chain(APADC.autoencoder1.parameters(), APADC.autoencoder2.parameters(), APADC.autoencoder3.parameters(), APADC.autoencoder4.parameters(), APADC.autoencoder5.parameters()),
                lr=config['training']['lr'])
            APADC.to_device(device)

            # Print the models
            # logger.info(APADC.autoencoder1)
            # logger.info(APADC.autoencoder2)
            # logger.info(optimizer)

            # Training
            acc, nmi, ari = APADC.train(config, logger, x1_train, x2_train, x3_train, x4_train, x5_train, Y_list, mask, optimizer, device)
            accumulated_metrics['acc'].append(acc)
            accumulated_metrics['nmi'].append(nmi)
            accumulated_metrics['ari'].append(ari)
            print('------------------------Training over------------------------')
        cal_std(logger, accumulated_metrics['acc'], accumulated_metrics['nmi'], accumulated_metrics['ari'])


if __name__ == '__main__':
    dataset = {0: "MNIST_USPS",
               1: "Caltech101-20",
               2: "RGB-D",
               3: "Scene-15",
               4: "NoisyMNIST",
               5: "ORL",
               6:"handwritten-5view",
               7:"MSRC_v1"}
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=int, default=str(7), help='dataset id')  # data index str(0)
    parser.add_argument('--test_time', type=int, default=str(5), help='number of test times')
    parser.add_argument('--devices', type=str, default='0', help='gpu device ids')
    args = parser.parse_args()
    dataset = dataset[args.dataset]
    MisingRate = [0.1, 0.3, 0.5, 0.7]
    if dataset == 'MSRC':
        MisingRate = [0.5]
    main(MR=MisingRate)
