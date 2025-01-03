import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-root', type=str, default='./data')
    parser.add_argument('-dataset', type=str, default='CAVE',
                        choices=['PaviaU','CAVE', 'Botswana', 'KSC', 'Urban', 'Pavia', 'IndianP', 'Washington','MUUFL_HSI','salinas_corrected',
                                 'Houston_HSI','chikusei'])
    parser.add_argument('--scale_ratio', type=float, default=4)
    parser.add_argument('--n_bands', type=int, default=31)
    parser.add_argument('--n_select_bands', type=int, default=3)

    parser.add_argument('--model_path', type=str,
                        default='./dataset_arch.pkl',
                        help='path for trained encoder')
    # parser.add_argument('--model_path', type=str,
    #                     default='./dataset_arch.pkl',
    #                     help='path for trained encoder')
    parser.add_argument('--train_dir', type=str, default='./data/dataset/train',
                        help='directory for resized images')
    parser.add_argument('--val_dir', type=str, default='./data/dataset/val',
                        help='directory for resized images')

    # learning settingl
    parser.add_argument('--n_epochs', type=int, default=20000,
                        help='end epoch for training')
    # rsicd: 3e-4, ucm: 1e-4,
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()
    return args
