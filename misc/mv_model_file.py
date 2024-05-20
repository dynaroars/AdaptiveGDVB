import sys
import os

folder = sys.argv[1]

files = [x for x in os.listdir(folder) if '.onnx' in x]

print(files)

for f in files:
    prefix = f[:-5]
    cmd1 = f'mkdir {folder}/{prefix}'
    cmd2 = f'mv {folder}/{f} {folder}/{prefix}'
    print(cmd1)
    print(cmd2)
    os.system(cmd1)
    os.system(cmd2)
    