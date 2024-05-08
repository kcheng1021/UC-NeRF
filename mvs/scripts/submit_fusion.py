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
    for scan in sets:
        command = f'''python multires.py -g inference_DTU -p 'get_data_loader.scan = "{scan}"' 'multires.output_folder = "{output_folder}/scan{scan}"'
python fusion.py -g inference_DTU -p 'get_data_loader.scan = "{scan}"' 'fusion.output_folder = "{output_folder}/scan{scan}"'
'''
        executor.name = f"{scan}_fusion"
        executor.submit(command)



    # Tanks and Temples jobs all sets
    sets = training_set + intermediate_set + advanced_set
    for scan in sets:
        command = f'''python multires.py -g inference_TNT -p 'get_data_loader.scan = "{scan}"' 'multires.output_folder = "{output_folder}/{scan}"'
python fusion.py -g inference_TNT -p 'get_data_loader.scan = "{scan}"' 'fusion.output_folder = "{output_folder}/{scan}"'
'''
        executor.name = f"{scan}_fusion"
        executor.submit(command)
