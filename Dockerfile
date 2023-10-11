FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
RUN pip install --upgrade "jax[cuda11_cudnn82]"==0.4.7 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && pip install neural-tangents==0.6.2
