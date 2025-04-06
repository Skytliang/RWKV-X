import os

def print_tree(path, indent=""):
    for item in os.listdir(path):
        full_path = os.path.join(path, item)
        print(f"{indent}- {item}")
        if os.path.isdir(full_path):
            print_tree(full_path, indent + "  ")

# 使用当前目录
print_tree(".")
