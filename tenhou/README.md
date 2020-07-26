# Tenhou

天鳳のmjlogからprotobufで読み込めるjsonへの変換は次のようにできる。

```sh
$ python mjlog_decoder.py resources/mjlog resources/json
```

また、protobufで読み込めるjsonから天鳳のmjlogへの変換は次のようにできる。

```sh
$ python mjlog_encoder.py resources/decoded_json resources/encoded_mjlog
```

天鳳のmjlogは、windows版の天鳳アプリを使って可視化することができる。
