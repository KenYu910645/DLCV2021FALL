import os 
import shutil

# Input 
intput_dir = "/home/lab530/KenYu/hw3-KenYu910645/hw3_data/p1_data/train/"
# Output
output_dir = "/home/lab530/KenYu/hw3-KenYu910645/hw3_data/p1_data_imagenet_format/train/"


for fn in os.listdir(intput_dir):
    label = fn.split("_")[0]
    
    try: 
        # Copy file 
        shutil.copyfile(intput_dir + fn, output_dir + label + '/' + fn)
    except FileNotFoundError as e :
        os.mkdir(output_dir + label + '/')
        shutil.copyfile(intput_dir + fn, output_dir + label + '/' + fn)


