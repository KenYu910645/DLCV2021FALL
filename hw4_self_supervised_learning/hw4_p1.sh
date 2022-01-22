# TODO: create shell script for running your prototypical network
# bash hw4_p1.sh hw4_data/mini/val.csv hw4_data/mini/val hw4_data/mini/val_testcase.csv tmp.csv
# python eval.py tmp.csv hw4_data/mini/val_testcase_gt.csv
# Example
python3 test_testcase.py --load hw4_p1_best.pth --test_csv $1 --test_data_dir $2 --testcase_csv $3 --output_csv $4 --matching_fn l2
