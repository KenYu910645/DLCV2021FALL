# DLCV-Fall-2021-Final-2

# How to run your code?
> TODO: Please provide example scripts to run your code. For example, 
> 1. python3 preprocessing.py <Path to data>
> 2. python3 inference.py <Path to the output csv file>
> ...

Please refer to [this link](https://docs.google.com/presentation/d/1775IaMakamj7jWtgZNY8T0166Pz1KXAUBCbzrvvr_0Y/edit?usp=sharing) for final project details and rules. **Note that all of final project videos and introduction pdf files can be accessed in your NTU COOL.**

## Usage
To start working on this project, you should clone this repository into your local machine by using the following command.
```bash
    git clone https://github.com/DLCV-Fall-2021/final-project-challenge-2-<username>.git
```
Note that you should replace `<username>` with your own GitHub username.

## Install Packages
To install the packages automatically, we have provided a `requirements.txt` file for this project. Please use the following script to set up the environment. Note that you are allowed to use any other Python library in this project.
```bash
pip3 install -r requirements.txt
```

## Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.
```bash
bash ./get_dataset.sh
```
The shell script will automatically download the dataset and store the data in a folder called `data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from the links in `get_dataset.sh` and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

## Evaluation Code
In the starter code of this repository, we have provided a python script for evaluating the results for this project. For Linux users, use the following command to evaluate the results of the sample submission.
```bash
python3 evaluation.py data/annotations.csv data/sampleSubmission.csv data/sample_seriesuids.csv
```

## Submission Rules
### Deadline
110/1/18 (Tue.) 23:59 (GMT+8)

## Q&A
If you have any problems related to the final project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under final project Discussion section in NTU COOL
