with open('image-gen-main (1).ipynb', 'r', encoding='utf-8') as f:
    content = f.read()

# Try replacing both variants just in case
content = content.replace(
    '!git clone https://github.com/HLE1236/airace --recursive',
    '!rm -rf /kaggle/working/gaussian-splatting\\n",\n    "!git clone https://github.com/HLE1236/airace gaussian-splatting --recursive'
)
content = content.replace(
    '!git clone https://github.com/fulx17/gaussian-splatting --recursive',
    '!rm -rf /kaggle/working/gaussian-splatting\\n",\n    "!git clone https://github.com/HLE1236/airace gaussian-splatting --recursive'
)

with open('image-gen-main (1).ipynb', 'w', encoding='utf-8') as f:
    f.write(content)
