FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git vim htop wget python-opencv libgl1-mesa-dev
RUN wget -O /tmp/anaconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh
RUN chmod +x /tmp/anaconda.sh
RUN sh -c /bin/echo -e "yes\n" | /tmp/anaconda.sh -b -p $HOME/anaconda3
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

RUN mkdir /workspace/notebooks
WORKDIR /workspace

RUN pip install --upgrade traitlets
RUN pip install --upgrade pygments

# Jupyter notebook
EXPOSE 8888

CMD [ "notebook", "--allow-root", "--no-browser" ]
ENTRYPOINT [ "jupyter" ]

# docker build -t bdas:latest
# docker run -d -v /home/kaisar:/home/kaisar bdas:latest
# INTERACTIVE
# docker run -it --shm-size=12GB --gpus all --network="host" -p 8888:8888 -v /home/kaisar:/home/kaisar --entrypoint /bin/bash bdas:latest
# DIRECTLY RUN
# docker run -it --shm-size=12GB --gpus all --network="host" -p 8888:8888 -v /home/kaisar/Documents/courses/BigDataSchoolAfrica/breast-mass-synthesis-tutorial-hidden:/workspace/notebooks bdas:latest
