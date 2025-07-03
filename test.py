import sys
sys.path.append('core')

import argparse
from PIL import Image
import torch
import torchvision.transforms as tf

from utils.flow_viz import flow_to_image
from utils.utils import InputPadder
from flowdiffuser import FlowDiffuser

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

transform = tf.Compose([
    tf.PILToTensor()
])


parser = argparse.ArgumentParser()
parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--dataset', help="dataset for evaluation")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args = parser.parse_args()
model = FlowDiffuser(args)
model.load_state_dict(torch.load('weights/FlowDiffuser-things.pth'), strict=False)
model.to(device)
model.eval()

with torch.no_grad():
    img0 = Image.open('sketch1.png').convert('RGB')
    img1 = Image.open('sketch3.png').convert('RGB')
    img0 = transform(img0)
    img1 = transform(img1)
    img0 = img0.to(device)
    img1 = img1.to(device)
    img0 = img0.unsqueeze(0)
    img1 = img1.unsqueeze(0)

    padder = InputPadder(img0.shape)
    img0, img1 = padder.pad(img0, img1)

    flow_low, flow_pr = model(img0, img1, iters=32, test_mode=True)
    flow = padder.unpad(flow_pr[0]).cpu()
    flow_img = flow_to_image(flow)
    flow_img.save('test.png')


