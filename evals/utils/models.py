import os
def get_latest_checkpoint(exp_path): 
    if exp_path.endswith(".pth"):
        # if the path is a checkpoint, return it
        return exp_path
    checkpoints = os.listdir(exp_path)
    checkpoints = [x for x in checkpoints if x.endswith(".pth")]
    checkpoints = sorted(checkpoints, key = lambda x: int(x.split("_")[1])) # sort by epochs 
    checkpoints = sorted(checkpoints, key = lambda x: int(x.split("_")[2].split(".")[0])) # sort by iterations
    # works because python sorting is stable
    return os.path.join(exp_path, checkpoints[-1])