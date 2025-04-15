import argparse
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import glob
import matplotlib
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2

from util.ana_utils import read_png_and_txt_ordered_cv2_rgb


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ANA Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--imagedata_root_dir', type=str, default='/home/ubuntu/Anantak/SensorUnit/data/Map/ImageData')
    parser.add_argument('--camera_num', type=int, default=0)
    parser.add_argument('--map_num', type=int, default=0)

    parser.add_argument('--output_root_dir', type=str, default='/home/ubuntu/Downloads/DepthAnythingMetric')
    
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr', default=True)

    args = parser.parse_args()
    
    if (args.map_num > 999):
        print("ERROR: args.map_num > 999")
        os.exit()
    
    if (args.map_num < 0):
        print("ERROR: args.map_num < 0")
        os.exit()

    if (args.camera_num > 99):
        print("ERROR: args.camera_num > 99")
        os.exit()
    
    if (args.camera_num < 0):
        print("ERROR: args.camera_num < 0")
        os.exit()

    # Create outputs
    args.input_images_dir = f"{args.imagedata_root_dir}/{int(args.map_num):03d}/{int(args.camera_num):02d}"
    args.outdir_1 = f"{args.output_root_dir}/{int(args.map_num):03d}"
    args.outdir = f"{args.output_root_dir}/{int(args.map_num):03d}/{int(args.camera_num):02d}"
    os.makedirs(args.output_root_dir, exist_ok=True)
    os.makedirs(args.outdir_1, exist_ok=True)
    os.makedirs(args.outdir, exist_ok=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # Load data
    #   frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
    filenames, timestamps = read_png_and_txt_ordered_cv2_rgb(args.input_images_dir)
    print(f"Read {len(filenames)} images from {args.input_images_dir}")
        
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        depth = depth_anything.infer_image(raw_image, args.input_size)
        
        if args.save_numpy:
            output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
            np.save(output_path, depth)

        if args.save_exr:
            output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.exr')
            # Ensure the array is float32
            if depth.dtype!= np.float32:
                print("Warning: Converting depth array to float32.")
                depth = depth.astype(np.float32)
            cv2.imwrite(output_path, depth)
            print(f"Written: {output_path}")

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
        if args.pred_only:
            cv2.imwrite(output_path, depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(output_path, combined_result)
            print(f"Written: {output_path}")
