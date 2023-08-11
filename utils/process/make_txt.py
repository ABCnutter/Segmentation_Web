import os


def make_txt(root, filename):
    path = os.path.join(root, filename)
    data = os.listdir(path)
    with open(r'E:\vs_code\pytorch\crack_seg\data\txt' + "/" + "train" + ".txt", "w") as f:
        for line in data:
            f.write(line + "\n")
        f.close()
        print("success!")


if __name__ == "__main__":
    root = r"E:\vs_code\pytorch\crack_seg\data\train"
    filename1 = "images"

    make_txt(root, filename1)