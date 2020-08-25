# Tenhou

天鳳のmjlogからprotobufで読み込めるjsonへの変換は次のようにできる。

```sh
$ python mjlog_decoder.py resources/mjlog resources/json
```

また、protobufで読み込めるjsonから天鳳のmjlogへの変換は次のようにできる。

```sh
$ python mjlog_encoder.py resources/json resources/restored_mjlog
```

天鳳のmjlogは、windows版の天鳳アプリを使って可視化することができる。

```sh
# python diff.py resources/mjlog resources/restored_mjlog
```

でencoderとdecoderの差分をチェックできる。
seedやdice等の情報は失われる。
