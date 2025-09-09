import os
import subprocess
import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity
from tqdm import tqdm
from natsort import natsorted

__LPIPS__ = {}
def init_lpips(net_name: str, device):
    assert net_name in ['alex', 'vgg']
    if net_name not in __LPIPS__:
        tqdm.write(f'init_lpips: lpips_{net_name}')
        __LPIPS__[net_name] = lpips.LPIPS(pretrained=True, net=net_name, version='0.1').eval().to(device)
    return __LPIPS__[net_name]
def compute_ssim(im1: np.ndarray, im2: np.ndarray):
    return structural_similarity(im1, im2, channel_axis=2)

def compute_lpips(im1: torch.Tensor, im2: torch.Tensor, model: str = 'alex'):
    if isinstance(im1, np.ndarray):
        im1 = torch.from_numpy(im1).permute([2, 0, 1]).contiguous().cuda()
    if isinstance(im2, np.ndarray):
        im2 = torch.from_numpy(im2).permute([2, 0, 1]).contiguous().cuda()
    device = im1.device
    lpips_model = init_lpips(model, device)
    return lpips_model(im1.unsqueeze(0), im2.unsqueeze(0), normalize=True).item()

def compute_psnr(im1: torch.Tensor, im2: torch.Tensor, mask: torch.Tensor = None):
    mse = torch.square(im1 - im2).mean(0).view(-1)
    if mask is not None:
        mse = mse[mask.reshape(-1) > 0]
    mse = mse.mean()
    return -10 * np.log10(mse.item())
def load_image(image_path):
    image_np = np.array(Image.open(image_path))[..., :3]
    image_pt = torch.from_numpy(image_np).permute([2, 0, 1])[:3] / 255.0
    return image_np, image_pt.cuda()

def evaluate(results_directory:str,output_directory:str,gt_folder:str):
    results = {}
    results["PSNR"] = []
    results["SSIM"] = []
    results["LPIPS"] = []
    from tqdm import tqdm
    gts = os.listdir(gt_folder)
    renders = os.listdir(results_directory)
    gts = natsorted(gts)
    renders = natsorted(renders)
    for i in tqdm(range(len(renders))):
        ours = os.path.join(results_directory, gts[i])
        gt = os.path.join(gt_folder, renders[i])
        groundtruth_np, groundtruth_pt = load_image(gt)
        prediction_np, prediction_pt = load_image(ours)
        results["PSNR"].append(compute_psnr(groundtruth_pt, prediction_pt, None))
        results["SSIM"].append(compute_ssim(groundtruth_np, prediction_np))
        results["LPIPS"].append(compute_lpips(groundtruth_pt, prediction_pt))
        averages = {metric: np.mean(values) for metric, values in results.items()}
        tqdm.write(f"== Evaluating with {len(results['PSNR'])} frames ==")
        for metric, average in averages.items():
            tqdm.write(f"{metric}: {average}")
    averages = {metric: np.mean(values) for metric, values in results.items()}
    for metric, average in averages.items():
        tqdm.write(f"{metric}: {average}")
    
    output_file_path = os.path.join(output_directory, "evaluation_results.txt")
    os.makedirs(output_directory, exist_ok=True)
    with open(output_file_path, "a") as file:
        file.write(f"Results directory: {results_directory}\n")
        for metric, average in averages.items():
            file.write(f"{metric}: {average}\n")
        # start frame and final frame
        file.write(f"{renders[0]} -- {renders[-1]}\n\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth", '-g', type=str, required=True)
    parser.add_argument("--result", '-r', type=str, required=True)
    parser.add_argument("--output_dir", '-o', type=str, default='.')
    args = parser.parse_args()
    evaluate(args.result,args.output_dir,args.groundtruth)

