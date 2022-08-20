import yaml

with open('items.yaml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    print(data)
    print(data['num'])

with open('items1.yaml', 'w') as f:
    yaml.dump(data, f)