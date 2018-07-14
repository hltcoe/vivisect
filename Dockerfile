FROM centos:7
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/root/.local/bin LANG=en_US.UTF-8

WORKDIR /root

RUN yum install epel-release -y && \
    yum install python36 git -y && \
    yum clean all -y && \
    curl https://bootstrap.pypa.io/get-pip.py | python36

COPY . /tmp/vivisect

WORKDIR /tmp/vivisect

RUN pip install . --user --pre --process-dependency-links -U

WORKDIR /root
