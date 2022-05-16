
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
# docker pull 


RUN if [ -f "/etc/apt/sources.list.d/cuda.list" ] ; then rm /etc/apt/sources.list.d/cuda.list ; fi
RUN if [ -f "/etc/apt/sources.list.d/nvidia-ml.list" ] ; then rm /etc/apt/sources.list.d/nvidia-ml.list ; fi

RUN apt-get update && \
    apt-get install -y build-essential  && \
    apt-get install -y wget && \
    apt-get install software-properties-common -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
RUN add-apt-repository universe

RUN apt-get update && apt-get install -y \
    apache2 \
    curl \
    git 


# Install miniconda

# RUN wget \
#     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh 

# RUN conda --version

ADD . /workspace/
COPY ./requirements.txt /workspace 
RUN pip install -r /workspace/requirements.txt 
# Still don't know why but we do need to install this separarelt at the end. It breaks everything if I don't do it. 
#  REMOVE AT YOUR OWN DISCRETION
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html 
# ENV PATH="/root/miniconda3/bin:${PATH}"
# SHELL ["/bin/bash", "-c"]

# RUN conda env create -n dp-adversarial --file /workspace/dp-adversarial/dp-adversarial.yml
# RUN conda init bash
# RUN echo "conda activate dp-adversarial" >> /root/.bashrc 
# RUN source /root/.bashrc
# RUN pip install -r /workspace/requirements.txt