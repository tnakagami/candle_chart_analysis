FROM jupyter/scipy-notebook:42f4c82a07ff

RUN conda install -y TA-Lib plotly mplfinance pytorch torchvision torchaudio cpuonly -c pytorch
