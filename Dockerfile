FROM nvcr.io/nvidia/l4t-pytorch:r-runtime

#Install all packages
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-tk

# install opencv ubuntu
RUN apt-get install python3-opencv
RUN apt-get install libpng-dev
RUN apt-get install libjpeg-dev
RUN apt-get install libopenexr-dev
RUN apt-get install libtiff-dev
RUN apt-get install libwebp-dev

# set working directory
WORKDIR /home/app

# Copy repo to working directory
ADD "Entry_Exit_REID" /home/app

# packages installation
RUN pip3 install -r "Entry_Exit_REID/requirements_deployment.txt"

CMD ["/bin/bash"]