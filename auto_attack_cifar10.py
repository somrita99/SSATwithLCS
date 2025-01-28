from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from models.wideresnet import *
from models.resnet import *
from utils import get_model
from autoattack import AutoAttack  # Import AutoAttack

parser = argparse.ArgumentParser(description='PyTorch CIFAR AutoAttack Evaluation')
parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                    help='input batch size for testing (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--model-path',
                    default='',
                    help='model for white-box attack evaluation')
parser.add_argument('--source-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='source model for black-box attack evaluation')
parser.add_argument('--target-model-path',
                    default='./checkpoints/model_cifar_wrn.pt',
                    help='target model for black-box attack evaluation')
parser.add_argument('--white-box-attack', default=True,
                    help='whether perform white-box attack')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

# set up data loader
transform_test = transforms.Compose([transforms.ToTensor(),])
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def eval_adv_test_autoattack(model, device, test_loader, epsilon):
    """
    Evaluate model by AutoAttack (white-box)
    """
    model.eval()
    total_samples = len(test_loader.dataset)

    # Initialize AutoAttack
    adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='standard')
    adversary.attacks_to_run = ['apgd-ce', 'apgd-dlr', 'fab-t', 'square']

    # Create data loader
    x_test, y_test = [], []
    for data, target in test_loader:
        x_test.append(data)
        y_test.append(target)
    x_test=x_test[:50]
    y_test=y_test[:50]
    x_test = torch.cat(x_test, 0).to(device)
    y_test = torch.cat(y_test, 0).to(device)
    total_samples = len(x_test)

    # Perform AutoAttack
    with torch.no_grad():
        adv_complete = adversary.run_standard_evaluation(x_test, y_test, bs=args.test_batch_size)
   # Print the keys and the shape of each tensor in adv_complete
    #print("Shape of adv_complete['clean_pred']: ", adv_complete['clean_pred'].shape)
    #print("adv_complete::", adv_complete['clean_pred'])
    #print("adv_complete::", adv_complete['adv_pred'])
    #print("clean_pred shape:", adv_complete['clean_pred'].shape if isinstance(adv_complete['clean_pred'], torch.Tensor) else type(adv_complete['clean_pred']))
    #print("adv_pred shape:", adv_complete['adv_pred'].shape if isinstance(adv_complete['adv_pred'], torch.Tensor) else type(adv_complete['adv_pred']))
    def flatten_pred(tensor):
        if tensor.dim() == 4:  # If it's batch x channels x height x width, reduce to batch x classes
            return tensor.argmax(dim=1)  # Get predicted class along channel dimension
        return tensor

    # Apply flattening to clean and adversarial predictions
    print(adv_complete)
    is_correct_adv = []

    # We only have one "iteration" in AutoAttack, but to maintain compatibility
    # with your PGD structure, we'll store the adversarial correctness once.
    is_correct_adv.append(np.reshape(
        (model(adv_complete).argmax(dim=1) == y).float().cpu().numpy(),
        [-1, 1])
    )

    # Convert the list of 2D arrays into a single 2D numpy array
    is_correct_adv = np.concatenate(is_correct_adv, axis=1)
    #clean_pred = flatten_pred(adv_complete['clean_pred'])
    #adv_pred = flatten_pred(adv_complete)
    #natural_accuracy = 100. * (total_samples - (adv_complete['clean_pred'] != y_test).float().sum().item()) / total_samples
    #robust_accuracy = 100. * (total_samples - (adv_complete != y_test).float().sum().item()) / total_samples
    print( is_correct_adv)
    print('AutoAttack (white-box):')
   # print('Natural Accuracy: {:.2f}%'.format(natural_accuracy))
    print('Robust Accuracy: {:.2f}%'.format(robust_accuracy))


def main():
    if args.white_box_attack:
        # white-box attack
        print('AutoAttack (white-box)')
        model = 'wrn-28-10'
        model = get_model(model, num_classes=10, normalize_input=False)
        if use_cuda:
            model = torch.nn.DataParallel(model).cuda()
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint)
        
        eval_adv_test_autoattack(model, device, test_loader, epsilon=args.epsilon)
    else:
        # black-box attack
        print('pgd black-box attack')
        model_target = WideResNet().to(device)
        model_target.load_state_dict(torch.load(args.target_model_path))
        model_source = WideResNet().to(device)
        model_source.load_state_dict(torch.load(args.source_model_path))

        eval_adv_test_blackbox(model_target, model_source, device, test_loader)

if __name__ == '__main__':
    main()
