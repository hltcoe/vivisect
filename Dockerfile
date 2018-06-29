FROM centos:7
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/.local/bin

WORKDIR /root

RUN yum install epel-release -y && \
    yum install python36 git -y && \
    yum clean all -y && \
    curl https://bootstrap.pypa.io/get-pip.py | python36

RUN git clone https://github.com/awslabs/sockeye.git && \
    git clone https://github.com/OpenNMT/OpenNMT-py.git && \
    cd sockeye && \
    pip install . --user --pre && \
    cd ../OpenNMT-py && \
    pip install . --user --pre && \
    cd .. && \
    rm -rf sockeye OpenNMT-py

COPY . /root/vivisect

WORKDIR /root/vivisect

RUN pip install . --user --pre

WORKDIR /root/vivisect

ENTRYPOINT python scripts/docker_entrypoint.sh
