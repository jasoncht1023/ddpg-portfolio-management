chcp 65001
rm requirements.txt
pip freeze > requirements.txt
sed -i "/torch==\|torchaudio==\|torchvision==/d" requirements.txt