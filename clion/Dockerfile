# from https://github.com/JetBrains/clion-remote/blob/master/Dockerfile.remote-cpp-env

FROM sotetsuk/ubuntu-gcc-grpc:v0.0.2

RUN apt-get update \
    && apt-get install -y ssh rsync texinfo

# install gdb-8.3
RUN wget https://ftp.gnu.org/gnu/gdb/gdb-8.3.tar.gz \
    && tar -xvzf gdb-8.3.tar.gz \
    && cd gdb-8.3 \
    && ./configure \
    && make -j4 \
    && make install

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd

RUN useradd -m user \
  && yes password | passwd user

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]
