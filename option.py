import argparse

parser = argparse.ArgumentParser(description='Low light image fusion')
# Seed
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Data Acquisition
parser.add_argument('--low_image_path', type=str, default='/home/yang/yang/data/LOLdataset/our485/low',
                    help='low light image path')
parser.add_argument('--high_image_path', type=str, default='/home/yang/yang/data/LOLdataset/our485/high',
                    help='groundtruth image path')
parser.add_argument('--test_lowimage_path', type=str, default='/home/yang/yang/data/LOLdataset/eval15/low',
                    help='low light image path')
parser.add_argument('--test_highimage_path', type=str, default='/home/yang/yang/data/LOLdataset/eval15/high',
                    help='groundtruth image path')

# Training
parser.add_argument('--enhance_batch', type=int, default=64, help='batch size of low light enhancement training')
parser.add_argument('--enhance_patch', type=int, default=96, help='patch size of low light enhancement training')
parser.add_argument('--enhance_epochs', type=int, default=4000, help='epochs of low light enhancement training')
parser.add_argument('--enhance_lr', type=float, default=1e-5, help='learning rate of low light enhancement training')

# Log file
parser.add_argument('--enhance_log_dir', type=str, default='./enhace_train_log', help='enhance training log file path')
parser.add_argument('--enhance_model_path', type=str, default='./enhance_model/', help='enhance model path')
parser.add_argument('--enhance_model', type=str, default='model.pth', help='enhance model name')

args = parser.parse_args()
