# Use Docker in CLion

次の手順でClionを使ってDockerコンテナ内でローカル環境に依存せずに開発ができる。
ここではポート2222番を使用する。

### 1. Dockerコンテナを立ち上げる

```sh
$ make docker-clion-start
```

この `make docker-clion-start` は次を行っている。

```sh
$ docker run -d --cap-add sys_ptrace -p 127.0.0.1:2222:22 --name mahjong-remote-clion sotetsuk/ubuntu-gcc-grpc-clion:latest
$ ssh-keygen -f "${HOME}/.ssh/known_hosts" -R "[localhost]:2222"
```

### 2. Toolchains からRemote Host を作る

Credentialsに "user@localhost:2222"を設定する。Passwordは "password"にする。Debuggerには/usr/local/bin/gdb を設定する。

![](https://user-images.githubusercontent.com/34413567/88379707-5e7e7100-cdde-11ea-9301-25625364516d.png)

### 3. CMake から新しくDebug環境を作る

Toolchainに2.で作ったRemote Hostを設定する。

![](https://user-images.githubusercontent.com/34413567/88379686-5b838080-cdde-11ea-88ba-36b2818b3123.png)

### 4. 新しく作ったデバッグ環境を選択する

![](https://user-images.githubusercontent.com/34413567/88380046-c9c84300-cdde-11ea-9457-9e038f17530e.png)

## References

- [Using Docker with CLion](https://blog.jetbrains.com/clion/2020/01/using-docker-with-clion/)
- [github.com/JetBrains/clion-remote](https://github.com/JetBrains/clion-remote)
