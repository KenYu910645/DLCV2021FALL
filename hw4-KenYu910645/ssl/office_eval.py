acc = 0
n_img = 0  
with open("../123.csv", 'r') as f:
    for i, line in enumerate(f.readlines()):
        if i != 0:
            n_img += 1
            _, fn, label = line.split('\n')[0].split(',')
            if label == fn[:-9]:
                acc += 1
print(f"Accurracy = {acc/n_img}")