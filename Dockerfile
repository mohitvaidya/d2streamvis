FROM nvidia/cuda:10.1-cudnn7-devel

ENV DEBIAN_FRONTEND noninteractive

# RUN  cp /etc/apt/sources.list ~/ && apt-get update && apt-get install -y wget && wget  "http://pastebin.com/raw.php?i=uzhrtg5M" -O /etc/apt/sources.list && \
#      apt-get update && rm /etc/apt/sources.list.d/ubuntu-extras.list && \
#      apt-get update

RUN cd /var/lib/apt/lists/ && rm -fr * && cd /etc/apt/sources.list.d/ && rm -fr * && cd /etc/apt && \
         cp sources.list sources.list.old &&  cp sources.list sources.list.tmp && \
         sed 's/ubuntuarchive.hnsdc.com/us.archive.ubuntu.com/' sources.list.tmp | tee sources.list &&\
          rm sources.list.tmp* && apt-get clean && apt-get update

RUN apt-get update && apt-get install -y \
    apache2 \
    curl \
    git \
    python3.7 \
    python3-pip
    
RUN apt-get update && apt-get install -y \
	python3-opencv ca-certificates wget sudo  \
	cmake ninja-build protobuf-compiler libprotobuf-dev 
# RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
# ARG USER_ID=1000
# RUN useradd -m --no-log-init --system  --uid ${USER_ID} appuser -g sudo
# RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# USER appuser
# WORKDIR /home/appuser

# ENV PATH="/home/appuser/.local/bin:${PATH}"
ENV PATH="/root/.local/bin:${PATH}"

RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py
# RUN apt-get update
# RUN apt-get install libav-tools
RUN apt-get install ffmpeg -y


# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install --user tensorboard
RUN pip install --user torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip install --user flask 

RUN pip install --user ffmpeg-python
RUN pip install --user azure-storage-blob==12.1.0 

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
# RUN sudo mkdir -p /app

# RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN mkdir /app
COPY source_code /app
COPY d2sourcecode /app


# ENV CUDA_HOME='/usr/local/cuda'


# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
# WORKDIR /app/detectron2

# WORKDIR /app

EXPOSE 8501

# ENTRYPOINT ["tail", "-f", "/dev/null"]

RUN pip install --user -r /app/requirements.txt
RUN pip install --user -e /app/detectron2
RUN pip install streamlit
RUN pip install numpy==1.16.4 llvmlite==0.33.0 --use-feature=2020-resolver



ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
WORKDIR /app

ENTRYPOINT ["streamlit", "run"]
CMD ["streammain.py"]

# ENTRYPOINT ["python3"]
# CMD ["/app/flask_app.py"]
