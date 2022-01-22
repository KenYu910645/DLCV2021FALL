wget -O hw3_best.pth https://www.dropbox.com/s/3ot2gxwv0goejhn/hw3_best.pth?dl=1
python3 p1_src/test.py --checkpoint hw3_best.pth --output_path $2 --input_dir $1