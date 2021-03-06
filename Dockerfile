FROM python:3.8

RUN apt update
RUN apt upgrade -y
RUN apt install -y libgl1-mesa-dev git

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN python -m pip install \
  jupyterlab numpy pandas scikit-learn requests cairosvg opencv-python pillow tqdm ipywidgets tensorflow==2.6.0 cloudpickle \
  matplotlib==3.4.3

RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py

ADD . /root

RUN git clone https://github.com/rishigami/Swin-Transformer-TF.git
