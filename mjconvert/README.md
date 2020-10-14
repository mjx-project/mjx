# Tenhou

天鳳のmjlogからprotobufで読み込めるjsonへの変換は次のようにできる。
mjproto側でのテストをしやすくするめに、細かい点を修正するためにmodifyオプションを設定する。

```sh
$ python mjlog_decoder.py resources/mjlog resources/json --modify
```

また、protobufで読み込めるjsonから天鳳のmjlogへの変換は次のようにできる。

```sh
$ python mjlog_encoder.py resources/json resources/restored_mjlog
```

天鳳のmjlogは、windows版の天鳳アプリを使って可視化することができる。


## Encoding/Decodingのテスト

次のようにmodifyオプションを外すことでしてEncoderとDecoderの差分をチェックできる。
seedやdice等の情報は失われる。

```sh
$ python mjlog_decoder.py resources/mjlog resources/json   # not --yaku-sorted
$ python mjlog_encoder.py resources/json resources/restored_mjlog
$ python diff.py resources/mjlog resources/restored_mjlog
```

## modifyオプション

Decoderのmodifyオプションがするのは次の修正:

1. 上がったときの役を役番号でソートする
2. 役満のときの符を常に0にする
