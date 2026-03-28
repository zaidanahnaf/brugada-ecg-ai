import json
with open('features/reduced/top_k_20_feature_names.json') as f:
    data = json.load(f)
print(type(data))
print(data)