1. 下载代码
   DiffusersTrain: git clone https://github.com/fzuzyb/DIffusersTran.git
2. 新建环境
    python3 -m venv local_env 
    source local_venv/bin/activate
3. 安装Transformers
    python3 -m setup develop -i https://pypi.tuna.tsinghua.edu.cn/simple
4. 安装accelerate
    python -m pip install accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
5. 安装Diffusers
    python -m setup develop -i https://pypi.tuna.tsinghua.edu.cn/simple
6. 安装PEFT
    python -m setup develop -i https://pypi.tuna.tsinghua.edu.cn/simple


7. 安装torchvision
   python -m pip install torchvision
   #python -m pip install torchvision==0.17.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
   python3 -m pip install opencv-python==4.9.0.80 -i https://pypi.tuna.tsinghua.edu.cn/simple
   python3 -m pip install scipy==1.12.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
   python3 -m pip install ipywidgets -i https://pypi.tuna.tsinghua.edu.cn/simple
   python3 -m pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
   python3 -m pip install xformers==0.0.20 -i https://pypi.tuna.tsinghua.edu.cn/simple
8. 安装ALGOHUB
    python -m setup develop -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install onnxruntime-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install PyCryptodome -i https://pypi.tuna.tsinghua.edu.cn/simple
    python -m pip install Cython
    python -m pip install einops
9.  python -m pip install bitsandbytes
9. Q&A
If you encounter a error dataset like "ValueError: Invalid pattern: '**' can only be an entire path component" for fsspec
   8. python -m pip install fsspec==2023.5.0
Data:
    Spider:
        python -m pip install retry
        python -m pip install jsonpath
        python -m pip install langid
    Process:
        python -m pip install scikit-learn

10.webdataset
pip install webdataset==0.2.86

# 高版本的torch
# 服务器
python -m pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install img2dataset



