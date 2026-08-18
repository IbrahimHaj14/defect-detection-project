import json
import pathlib

# Locate raw SSGD folder
possible_roots = [
    pathlib.Path('data/raw/SSGD'),
    pathlib.Path('data/raw/SSGD-Dataset'),
    pathlib.Path('data/raw'),
]

raw_root = None
for p in possible_roots:
    if p.exists() and any(p.iterdir()):
        raw_root = p
        break

if not raw_root:
    print("Could not locate raw SSGD folder under data/raw/")
else:
    print(f"=== Auditing Raw Dataset at: {raw_root} ===")
    
    # 1. Inspect image files in raw
    all_files = list(raw_root.rglob('*.*'))
    images = [p for p in all_files if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
    json_files = [p for p in all_files if p.suffix.lower() == '.json']
    
    print(f"Total raw image files found: {len(images)}")
    print(f"Total JSON annotation files found: {len(json_files)}")
    
    # 2. Inspect JSON contents
    for jf in json_files:
        try:
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                images_in_json = len(data.get('images', []))
                anns_in_json = len(data.get('annotations', []))
                keys = list(data.keys())
                print(f"\nJSON: {jf.relative_to(raw_root)}")
                print(f"  Keys: {keys}")
                if 'images' in data:
                    print(f"  Image entries in JSON: {images_in_json}")
                if 'annotations' in data:
                    print(f"  Annotation entries in JSON: {anns_in_json}")
            elif isinstance(data, list):
                print(f"\nJSON: {jf.relative_to(raw_root)} -> List with {len(data)} items")
        except Exception as e:
            print(f"Could not read {jf.name}: {e}")
