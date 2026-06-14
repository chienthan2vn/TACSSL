import os
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import Counter

def main():
    base_dir = Path("/media/merlin/New Volume/project/kltn/dataset/src_data")
    plantdoc_dir = base_dir / "plantdoc"
    plantwild_dir = base_dir / "plantwild"
    plantseg_dir = base_dir / "plantseg"
    target_dir = base_dir / "ssl_data"

    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get all plantdoc classes
    plantdoc_classes = set()
    for split in ['train', 'test']:
        split_dir = plantdoc_dir / split
        if split_dir.exists():
            for class_name in os.listdir(split_dir):
                if (split_dir / class_name).is_dir():
                    plantdoc_classes.add(class_name.lower())

    # 2. Function to check if plantwild class should be included
    def should_include_plantwild_class(pw_class):
        pw_class_lower = pw_class.lower().replace('_', ' ')
        first_word = pw_class_lower.split()[0] if pw_class_lower else ""
        if not first_word:
            return False
        
        for pd_class in plantdoc_classes:
            if first_word in pd_class.replace('_', ' '):
                return True
        return False

    # 3. Copy plantdoc data
    print("Gathering PlantDoc data...")
    plantdoc_tasks = []
    plantdoc_stats = Counter()
    for split in ['train', 'test']:
        split_dir = plantdoc_dir / split
        if not split_dir.exists(): continue
        for class_name in os.listdir(split_dir):
            class_dir = split_dir / class_name
            if not class_dir.is_dir(): continue
            
            for img_path in class_dir.glob("*"):
                if img_path.is_file():
                    new_name = f"plantdoc_{split}_{class_name.replace(' ', '_')}_{img_path.name}"
                    plantdoc_tasks.append((img_path, target_dir / new_name))
                    plantdoc_stats[class_name] += 1

    print(f"Copying {len(plantdoc_tasks)} PlantDoc images...")
    for src, dst in tqdm(plantdoc_tasks, desc="PlantDoc"):
        shutil.copy2(src, dst)

    # 4. Copy plantwild data
    print("Gathering PlantWild data...")
    plantwild_tasks = []
    included_pw_classes = set()
    plantwild_stats = Counter()
    
    for split in ['train', 'val', 'test']:
        split_dir = plantwild_dir / split
        if not split_dir.exists(): continue
        for class_name in os.listdir(split_dir):
            class_dir = split_dir / class_name
            if not class_dir.is_dir(): continue
            
            if should_include_plantwild_class(class_name):
                included_pw_classes.add(class_name)
                for img_path in class_dir.glob("*"):
                    if img_path.is_file():
                        new_name = f"plantwild_{split}_{class_name.replace(' ', '_')}_{img_path.name}"
                        plantwild_tasks.append((img_path, target_dir / new_name))
                        plantwild_stats[class_name] += 1

    print(f"Selected PlantWild classes: {sorted(list(included_pw_classes))}")
    print(f"Copying {len(plantwild_tasks)} PlantWild images...")
    for src, dst in tqdm(plantwild_tasks, desc="PlantWild"):
        shutil.copy2(src, dst)

    # 5. Copy plantseg data
    print("Gathering PlantSeg data...")
    plantseg_tasks = []
    included_ps_classes = set()
    plantseg_stats = Counter()

    # 2. Function to check if plantseg class should be included
    def should_include_plantseg_class(ps_class):
        ps_class_lower = ps_class.lower().replace('_', ' ')
        first_word = ps_class_lower.split()[0] if ps_class_lower else ""
        if not first_word:
            return False
        
        for pd_class in plantdoc_classes:
            if first_word in pd_class.replace('_', ' '):
                return True
        return False
    
    for split in ['train', 'val', 'test']:
        split_dir = plantseg_dir / split
        if not split_dir.exists(): continue
        for class_name in os.listdir(split_dir):
            class_dir = split_dir / class_name
            if not class_dir.is_dir(): continue
            
            if should_include_plantseg_class(class_name):
                included_ps_classes.add(class_name)
                for img_path in class_dir.glob("*"):
                    if img_path.is_file():
                        new_name = f"plantseg_{split}_{class_name.replace(' ', '_')}_{img_path.name}"
                        plantseg_tasks.append((img_path, target_dir / new_name))
                        plantseg_stats[class_name] += 1

    print(f"Selected PlantSeg classes: {sorted(list(included_ps_classes))}")
    print(f"Copying {len(plantseg_tasks)} PlantSeg images...")
    for src, dst in tqdm(plantseg_tasks, desc="PlantSeg"):
        shutil.copy2(src, dst)

    print(f"\nDone! Copied total {len(plantdoc_tasks) + len(plantwild_tasks) + len(plantseg_tasks)} images to {target_dir}")

    print("\n" + "="*50)
    print("STATISTICS: NUMBER OF IMAGES PER CLASS")
    print("="*50)
    
    print(f"\n[PlantDoc] Total: {sum(plantdoc_stats.values())} images across {len(plantdoc_stats)} classes")
    for class_name, count in plantdoc_stats.most_common():
        print(f"  - {class_name}: {count}")
        
    print(f"\n[PlantWild] Total: {sum(plantwild_stats.values())} images across {len(plantwild_stats)} classes")
    for class_name, count in plantwild_stats.most_common():
        print(f"  - {class_name}: {count}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
