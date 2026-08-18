# 🎬 Movie Recommender — PyTorch Two-Tower Neural Network

## 📝 How to perform performance measurement on Intel B70? (Dnotitia's VP Team)

** Please keep following steps. Target environment is Ubuntu 24.04.04 and Kernel 6.17.0-42-generic.

** Install Intel Compute Runtime
- wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | sudo gpg --yes --dearmor -o /usr/share/keyrings/intel-graphics.gpg
- echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" | sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list && sudo apt update
- sudo apt install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo

** Install PyTorch XPU
- python3 -m venv .venv && .venv/bin/pip install --upgrade pip
- .venv/bin/pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/xpu
- .venv/bin/pip install pandas==2.3.3 numpy==2.3.5 pyarrow tqdm
- .venv/bin/python src/check_torch_install.py

** How to Measure?
- .venv/bin/python src/perf_eval.py --batches 1 8 32 64 128 256 512 1024 2048
- .venv/bin/python src/perf_eval.py --devices xpu --torch-threads N --out result.json

## 📝 License

## 📝 Detailed Information
- My repositorty is forked from https://github.com/nickgreenquist/Movie-Recommender-System-PyTorch-TwoTower-Model. Please check detailed information in the main repository.

[MIT](LICENSE) © Nick Greenquist
</content>
