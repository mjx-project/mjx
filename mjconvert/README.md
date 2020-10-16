# mjconvert

## インストール

```sh
$ make install
```

## 使い方

天鳳のmjlogからprotobufで読み込めるjsonへの変換は次のようにできる。
mjproto側でのテストをしやすくするめに、細かい点が修正される。

```sh
$ mjconvert resources/mjlog resources/json --to-mjproto
```

また、protobufで読み込めるjsonから天鳳のmjlogへの変換は次のようにできる。

```sh
$ mjconvert resources/json resources/restored_mjlog --to-mjlog
```

天鳳のmjlogは、windows版の天鳳アプリを使って可視化することができる。


## Encoding/Decodingのテスト

次のように `--to-mjproto-raw` でしてEncoderとDecoderの差分をチェックできる。
seedやdice等の情報は失われる。

```sh
$ mjconvert resources/mjlog resources/json --to-mjproto-raw
$ mjconvert resources/json resources/restored_mjlog --to-mjlog
$ python diff.py resources/mjlog resources/restored_mjlog
```

## 変換において失われる情報

- mjlog => mjproto-raw: seed, dice, 接続切れ
- mjlog => mjproto: mjproto-rawに加え、(1) 上がったときの役が役番号でソートされる (2) 役満のときの符が常に0にセットされる
- mjproto => mjlog
