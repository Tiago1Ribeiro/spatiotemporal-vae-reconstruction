import yaml

# read the config file
with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# get the input size
input_size = config["data"]["input_size"]

# get the first element of the tuple
first_element = input_size[0]

print(first_element)  # prints: 512