import psutil

def print_memory_usage(label):
    mem = psutil.virtual_memory()
    print(f"({label}: Total={mem.total/1e9:.2f} GB, Available={mem.available/1e9:.2f} GB, Used={mem.used/1e9:.2f} GB, Percent={mem.percent}%)\n")
