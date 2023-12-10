import argparse
import os

import gin
from submitter import submitter

# DTU
val_set = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]
test_set = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77, 110, 114, 118]

# TNT
training_set = ['Barn', 'Truck', 'Caterpillar', "Ignatius", 'Meetingroom', 'Church', 'Courthouse']
intermediate_set = ['Family','Francis','Horse','Lighthouse','M60','Panther','Playground','Train']
advanced_set = ["Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple"]

output_folder = "results"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gin_config', nargs='+', default=['submitter'],
                        help='Set of config files for gin (separated by spaces) '
                        'e.g. --gin_config file1 file2 (exclude .gin from path)')
    parser.add_argument('-p', '--gin_param', nargs='+', default=[],
                        help='Parameter settings that override config defaults '
                        'e.g. --gin_param module_1.a=2 module_2.b=3')
    args = parser.parse_args()
    gin_files = [f'configs/{g}.gin' for g in args.gin_config]
    gin.parse_config_files_and_bindings(
        gin_files, args.gin_param, skip_unknown=True)

    executor = submitter()

    # DTU jobs both val/test set
    sets = val_set + test_set
    for scale, nf in [(1, 10), (2, 10)]:
        for scan in sets:
            command = f'''python inference.py -g inference_DTU -p \\
'inference.scan = "scan{scan}"' \\
'inference.num_frame = {nf}' \\
'inference.rescale = {scale}'
'''
            executor.name = f"{scan}_{scale}_{nf}"
            executor.submit(command)



    # Tanks and Temples jobs all sets
    sets = training_set + intermediate_set + advanced_set
    for scale, nf in [(1, 15), (2, 25)]:
        for scan in sets:
            if scan in training_set:
                image_folder = f"datasets/TanksAndTemples/training_input/{scan}/images"
            elif scan in intermediate_set:
                image_folder = f"datasets/TanksAndTemples/tankandtemples/intermediate/{scan}/images"
            else:
                image_folder = f"datasets/TanksAndTemples/tankandtemples/advanced/{scan}/images"
            images = sorted(os.listdir(image_folder))
            N = len(os.listdir(image_folder))
            # group every 100 images together
            step = 100
            startend = []
            for i in range(0, N, step):
                startend.append((i, min(N, i + step)))

            for start, end in startend:

                command = f'''python inference.py -g inference_TNT -p \\
'get_data_loader.scan = "{scan}"' \\
'get_data_loader.num_frame = {nf}' \\
'get_data_loader.subset = ({start}, {end}, 1)' \\
'inference.rescale = {scale}' \\
'inference.output_folder = "{output_folder}/{scan}"'
'''
                executor.name = f"{scan}_{scale}_{nf}_{start}"
                executor.submit(command)
