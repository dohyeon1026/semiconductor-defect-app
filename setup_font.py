import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import shutil

# 폰트 다운로드 경로
font_url = "https://github.com/naver/nanumfont/blob/master/ttf/NanumGothic.ttf?raw=true"
font_path = "/tmp/NanumGothic.ttf"

# 다운로드
import urllib.request
urllib.request.urlretrieve(font_url, font_path)

# 폰트 등록
font_dir = "/root/.fonts"
os.makedirs(font_dir, exist_ok=True)
shutil.copy(font_path, font_dir)

# Matplotlib에 등록
fm.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "NanumGothic"

