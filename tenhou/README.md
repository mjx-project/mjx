# Tenhou

天鳳のmjlogからprotobufで読み込めるjsonへの変換は次のようにできる。
天鳳での上がり時の役の並び方はトポロジカルにソートできず、再現が難しそうなため、役の番号でソートしている。

```sh
$ python mjlog_decoder.py resources/mjlog resources/json --yaku-sorted
```

また、protobufで読み込めるjsonから天鳳のmjlogへの変換は次のようにできる。

```sh
$ python mjlog_encoder.py resources/json resources/restored_mjlog
```

天鳳のmjlogは、windows版の天鳳アプリを使って可視化することができる。


## Encoding/Decodingのテスト

次のようにしてEncoderとDecoderの差分をチェックできる。
seedやdice等の情報は失われる。
ここでのチェックでは役はソートしない。

```sh
$ python mjlog_decoder.py resources/mjlog resources/json   # not --yaku-sorted
$ python mjlog_encoder.py resources/json resources/restored_mjlog
$ python diff.py resources/mjlog resources/restored_mjlog
```

