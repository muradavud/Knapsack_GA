import random

with open("matrix_1000.txt", "w+") as f:
    for i in range(1000):
        for i in range(1000):
            f.write(str(round(random.uniform(-1, 2), 6)) + "  ")
        f.write("\n")
    f.close()

with open("vector_1000.txt", "w+") as f:
    for i in range(1000):
        f.write(str(round(random.uniform(-10000, 10000), 3)) + "  ")
        f.write("\n")
    f.close()
