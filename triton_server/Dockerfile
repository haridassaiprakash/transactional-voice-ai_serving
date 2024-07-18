FROM nvcr.io/nvidia/tritonserver:23.07-py3
RUN apt-get update && apt-get install -y libsndfile1 build-essential libssl-dev swig
RUN pip3 install --no-cache-dir pyctcdecode webrtcvad setuptools==59.5.0 transformers torch --extra-index-url https://download.pytorch.org/whl/cu11
RUN cd /tmp && wget https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-x86_64.tar.gz &&\
    tar -zxvf cmake-3.25.1-linux-x86_64.tar.gz 
RUN git clone https://github.com/nvidia-riva/riva-asrlib-decoder &&\
    cd riva-asrlib-decoder && git submodule update --init --recursive &&\
    mkdir build && cd build &&\
    /tmp/cmake-3.25.1-linux-x86_64/bin/cmake -DRIVA_ASRLIB_BUILD_PYTHON_BINDINGS=NO .. &&\
    make -j 2
COPY scripts/swig /home/swig
RUN cd /home/swig && bash setup.sh
RUN python3 -m pip install git+https://github.com/huggingface/optimum.git#egg=optimum[onnxruntime-gpu] 
RUN pip install --no-cache-dir --force-reinstall "faster-whisper @ https://github.com/guillaumekln/faster-whisper/archive/3adcc12d0f91369446a624e33185c555facc8ed2.tar.gz" joblib charset-normalizer==3.1.0
RUN pip uninstall -y onnxruntime && pip install --no-cache-dir onnxruntime-gpu==1.13.1 numpy==1.23.1
RUN pip install cython
RUN pip install nemo_toolkit['all']==1.21.0
RUN git clone https://github.com/haridassaiprakash/indic-punct.git && cd indic-punct && git checkout main && bash install.sh && pip install --no-cache-dir pynini && python3 setup.py bdist_wheel && pip install -e .
RUN pip install pyyaml accelerate protobuf https://github.com/kpu/kenlm/archive/master.zip transformers -U
RUN pip install protobuf==3.20 
ARG INDICPUNCT_COMMIT_HASH
RUN cd indic-punct && git pull && git checkout $INDICPUNCT_COMMIT_HASH
RUN mkdir /opt/tritonserver/logs
CMD tritonserver --model-repository=/models/model_repository
EXPOSE 8001
EXPOSE 8002
EXPOSE 8003