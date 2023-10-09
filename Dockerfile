# Caution: Python version >= 3.7 (seaborn package prerequisites)
#           for version 3.6.9 it can be used but have to install seaborn via "sudo apt-get install pythoon3-seaborn"

FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3

#Install all packages
RUN apt-get update && apt-get upgrade -y

RUN apt install apt-utils
RUN apt-get install sudo -y
RUN sudo apt-get install python3-pip -y
RUN apt-get install python3-tk -y
RUN apt-get install python3-seaborn
RUN pip3 uninstall pandas -y
RUN apt-get install python3-pandas -y

# packages installation
RUN pip3 install gdown
RUN pip3 install lap
RUN pip3 install cython_bbox

RUN cd deep-person-reid && \
    python3 setup.py develop \
    cd ..

#RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-tk

# set working directory
WORKDIR /home/app

# Copy repo to working directory
ADD "Entry_Exit_REID" /home/app

#remaining packages will be auto installed by setup.py deep person reid in the next steps
# RUN cd deep-person-reid && python3 setup.py develop
# RUN cd ..

CMD ["/bin/bash"]