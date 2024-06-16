import os

def getFiles(targetdir):
    ls = []
    for fname in os.listdir(targetdir):
        path = os.path.join(targetdir, fname)
        if os.path.isdir(path):
            continue
        ls.append(fname)
    return sorted(ls)

def getDirs(parent_dir):
    ls = []
    for dir_name in os.listdir(parent_dir):
        path = os.path.join(parent_dir, dir_name)
        if os.path.isdir(path):
            ls.append(dir_name)
    return sorted(ls)   