# Use Docker in CLion

1. コンテナを作る.
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
Debuggerには/usr/local/bin/gdb を設定する.

![](https://user-images.githubusercontent.com/34413567/88379707-5e7e7100-cdde-11ea-9301-25625364516d.png)

3. CMake から新しくDebug環境を作り, 
Toolchainに2.で作ったRemote Hostを設定する.
![](https://user-images.githubusercontent.com/34413567/88379686-5b838080-cdde-11ea-88ba-36b2818b3123.png)

4. 新しく作ったデバッグ環境を選択する.
![](https://user-images.githubusercontent.com/34413567/88380046-c9c84300-cdde-11ea-9457-9e038f17530e.png)
