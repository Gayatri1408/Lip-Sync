# -*- coding: utf-8 -*-
"""Task-1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SwPVG9jYPsfuM0lgbvYtIfBN1gjiHyRK
"""

from google.colab import drive
drive.mount('/content/gdrive')

!git clone https://github.com/Gayatri1408/Lip-Sync.git

!ls

!cp -ri "/content/gdrive/MyDrive/Projects/wav2lip/weights/wav2lip_gan.pth" /content/Wav2Lip/checkpoints/

!pip uninstall tensorflow tensorflow-gpu

!cd Wav2Lip && pip install -r requirements.txt

!wget "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O "Wav2Lip/face_detection/detection/sfd/s3fd.pth"

!cp "/content/gdrive/MyDrive/Projects/wav2lip/input/Task_VIDEO.mp4" "/content/gdrive/MyDrive/Projects/wav2lip/input/output10.wav" sample_data/
!ls sample_data/

!cd Wav2Lip && python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face "/content/gdrive/MyDrive/Projects/wav2lip/input/Task_VIDEO.mp4" --audio "/content/gdrive/MyDrive/Projects/wav2lip/input/output10.wav" --resize_factor 2