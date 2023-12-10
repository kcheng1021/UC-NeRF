import argparse
from pathlib import Path

import gin

from datasets import get_test_data_loader
from fusion import fusion
from inference import inference
from multires import multires

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gin_config', nargs='+', default=[],
                        help='Set of config files for gin (separated by spaces) '
                        'e.g. --gin_config file1 file2 (exclude .gin from path)')
    parser.add_argument('-p', '--gin_param', nargs='+', default=[],
                        help='Parameter settings that override config defaults '
                        'e.g. --gin_param module_1.a=2 module_2.b=3')
    args = parser.parse_args()
    gin_files = [f'configs/{g}.gin' for g in args.gin_config]
    gin.parse_config_files_and_bindings(
        gin_files, args.gin_param, skip_unknown=True)

    output_folder = Path("results")

    # demo for DTU
    demo_scans = ["scan3"]
    for scan in demo_scans:
        # Low Res and High Res pass
        for rescale, num_frames in [(1, 10), (2, 10)]:
            data_loader = get_test_data_loader("DTUTest", scan=scan, num_frames=num_frames)
            inference(
                data_loader,
                ckpt="pretrained/train_DTU.pth",
                output_folder=output_folder / scan,
                rescale=rescale,
                do_report=True
            )
        # Multi Res Fusion
        multires(output_folder / scan, suffix1="_nf10", suffix2="_nf10", visualize=True)
        # Adaptive Threshold Fusion
        data_loader = get_test_data_loader("DTUTest", scan=scan, num_frames=10)
        fusion(data_loader, output_folder / scan, rescale=2, suffix="_nf10_nf10_th0.02")

    # demo for TNT
    demo_scans = ["Ignatius", "Meetingroom"]
    for scan in demo_scans:
        # Low Res and High Res pass
        for rescale, num_frames in [(1, 15), (2, 25)]:
            data_loader = get_test_data_loader("TNT", scan=scan, num_frames=num_frames)
            inference(
                data_loader,
                ckpt="pretrained/train_BlendedMVS.pth",
                output_folder=output_folder / scan,
                rescale=rescale,
                do_report=True
            )
        # Multi Res Fusion
        multires(output_folder / scan, suffix1="_nf15", suffix2="_nf25", visualize=True)
        # Adaptive Threshold Fusion
        data_loader = get_test_data_loader("TNT", scan=scan, num_frames=10)
        fusion(data_loader, output_folder / scan, rescale=1, suffix="_nf15_nf25_th0.02")
