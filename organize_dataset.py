import os
import shutil

label_file = r"LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"
audio_folder = r"LA\ASVspoof2019_LA_train\flac"

real_folder = r"dataset\real"
fake_folder = r"dataset\fake"

os.makedirs(real_folder, exist_ok=True)
os.makedirs(fake_folder, exist_ok=True)

count = 0

with open(label_file, "r") as f:

    for line in f:

        parts = line.strip().split()

        audio_id = parts[1]
        label = parts[-1]

        file_name = audio_id + ".flac"

        source = os.path.join(audio_folder, file_name)

        if os.path.exists(source):

            if label == "bonafide":
                dest = os.path.join(real_folder, file_name)
            else:
                dest = os.path.join(fake_folder, file_name)

            shutil.copy(source, dest)

            count += 1

print("Files copied:", count)
print("Dataset organized successfully!")