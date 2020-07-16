# Use Docker in CLion

1. コンテナを作る 
```shell script
$ docker build -t clion/remote-cpp-env:0.5 .  
$ docker run -d --cap-add sys_ptrace -p127.0.0.1:2222:22 \
      --name clion_remote_env clion/remote-cpp-env:0.5
$ ssh-keygen -f "$HOME/.ssh/known_hosts" -R "[localhost]:2222"
$ ssh user@localhost -p2222
    # yes
    # password
```

2. Toolchains からRemote Host を作り,
Credentialsに "user@localhost:2222"を設定する
(Passwordは "password"にする).

3. CMake から新しくDebug環境を作り, 
Toolchainに2.で作ったRemote Hostを設定する.
