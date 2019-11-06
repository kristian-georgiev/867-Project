# to load the yaml file
import ruamel.yaml as yaml

def parse(filename):
    print(f"Parsing {filename}")
    with open(filename) as stream:
        try:
            return yaml.safe_load(stream)

        except yaml.YAMLError as exc:
            return exc