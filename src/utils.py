import yaml

def read_config(filename: str):
    with open(filename, "r") as file:
        config = yaml.safe_load(file)
        return config
    
def format_number(num):
    if num < 1000:
        return str(num)
    elif num < 1000000:
        return f"{num/1000:.1f}k"
    else:
        return f"{num/1000000:.1f}m"