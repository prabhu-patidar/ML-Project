import os
import shutil

path = "flowers"

for file in os.listdir(path):
    file_path = os.path.join(path, file)

    if os.path.isdir(file_path):
        continue

    class_name = file.split("_")[0]

    class_folder = os.path.join(path, class_name)

    if not os.path.exists(class_folder):
        os.makedirs(class_folder)

    shutil.move(file_path, os.path.join(class_folder, file))

print("Images organized successfully!")