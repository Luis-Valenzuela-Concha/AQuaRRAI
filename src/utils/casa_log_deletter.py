import os

def delete_casa_logs(path="."):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")
    os.system(f"rm -rf {path}/casa-*.log")
    