import sys
sys.path.append('core')

import argparse
import torch
import cv2

from utils.flow_viz import flow_to_image
from utils.frame_utils import writeFlow
from utils.utils import InputPadder
from flowdiffuser import FlowDiffuser

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def to_tensor(img, device=device):
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)


parser = argparse.ArgumentParser()
parser.add_argument('--model', help="restore checkpoint")
parser.add_argument('--dataset', help="dataset for evaluation")
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
args = parser.parse_args()
model = torch.nn.DataParallel(FlowDiffuser(args))
model.load_state_dict(torch.load('weights/FlowDiffuser-things.pth'))
model.to(device)
model.eval()

with torch.no_grad():
    img0 = cv2.imread('frame1.png')
    img1 = cv2.imread('frame2.png')
    img0 = to_tensor(img0)
    img1 = to_tensor(img1)

    padder = InputPadder(img0.shape)
    img0, img1 = padder.pad(img0, img1)

    flow_low, flow_pr = model(img0, img1, iters=32, test_mode=True)
    flow = flow_pr.squeeze(0)
    flow = padder.unpad(flow)
    flow = flow.permute(1, 2, 0).cpu().numpy()
    rgb = flow_to_image(flow)
    cv2.imwrite('test.png', rgb)

