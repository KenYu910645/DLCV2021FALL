with open('training.txt', 'w') as f:
    s = ""
    for i in range(40):
        for j in range(9):
            s += str(i+1) + "_" + str(j+1) + ".png\n"
    f.write(s)

with open('testing.txt', 'w') as f:
    s = ""
    for i in range(40):
        s += str(i+1) + "_10.png\n"
    f.write(s)