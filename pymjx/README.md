# mjx

## インストール

```sh
$ make install
```

## 使い方

天鳳のmjlogからprotobufで読み込めるjsonへの変換は次のようにできる。
mjxproto側でのテストをしやすくするめに、細かい点が修正される。

```sh
$ mjx resources/mjlog resources/json --to-mjxproto
```

また、protobufで読み込めるjsonから天鳳のmjlogへの変換は次のようにできる。

```sh
$ mjx resources/json resources/restored_mjlog --to-mjlog
```

天鳳のmjlogは、windows版の天鳳アプリを使って可視化することができる。


## Encoding/Decodingのテスト

次のように `--to-mjxproto-raw` でしてEncoderとDecoderの差分をチェックできる。
seedやdice等の情報は失われる。

```sh
$ mjx resources/mjlog resources/json --to-mjxproto-raw
$ mjx resources/json resources/restored_mjlog --to-mjlog
$ python diff.py resources/mjlog resources/restored_mjlog
```

## 変換において変わってしまう・失われてしまう情報

- mjlog => mjxproto-raw: seed, dice, 接続切れ
- mjlog => mjxproto: mjxproto-rawに加え、(1) 上がったときの役が役番号でソートされる (2) 役満のときの符が常に0にセットされる
- mjxproto => mjlog: ユーザ名の％エンコーディング
