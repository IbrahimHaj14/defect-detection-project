import json
import pathlib

json_dir = pathlib.Path('data/raw/ECF-Dataset/Serious defect/ir_base/train_NG_labeled/damper_missing')
orphaned_jsons = ['img_13f8bea6.json', 'img_31087c30.json', 'img_79ce870b.json', 'img_ece1ca9.json']

print('=== Checking imagePath inside orphaned JSONs ===')
for j_name in orphaned_jsons:
    j_path = json_dir / j_name
    if j_path.exists():
        with open(j_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        img_path = data.get('imagePath', 'N/A')
        print(f"JSON: {j_name:<20} | imagePath inside: {img_path}")
