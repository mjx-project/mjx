# Open Mahjong

[![](./img/architecture.png)](https://docs.google.com/presentation/d/1lhb_sNix02Iyp0DI0Be5uuQub1W7CVbFiUdiDazG6tY/edit?usp=sharing)

## Tenhou/mjlog
天鳳から天鳳位などの牌譜をダウンロードすると、基本的に `.mjlog` という拡張子のファイルが得られる。これはそのままWindows版の天鳳で観戦することができる。このファイルの中身をテキストエディタ等で直接編集したい場合は、一度 `.gz` 形式になおしてから解凍する必要がある[[1](http://rausumaru.hatenablog.com/entry/2019/08/30/021154)]。名前の匿名表示のオンオフはプレミアム版で設定可能らしい（要確認）。

## Terms
| Japanese | English | Reference |
|:---:|:---:|:---:|
| チー | Chi  | [Japanese Mahjong (Wikipedia)](https://en.wikipedia.org/wiki/Japanese_Mahjong) |
| ポン | Pon  | [Japanese Mahjong (Wikipedia)](https://en.wikipedia.org/wiki/Japanese_Mahjong) |
| カン | Kan  | [Japanese Mahjong (Wikipedia)](https://en.wikipedia.org/wiki/Japanese_Mahjong) |

## References

- Hand score calculation
  - [シャンテン数計算アルゴリズム](https://qiita.com/tomo_hxx/items/75b5f771285e1334c0a5)
    - [github.com/tomohxx/shanten-number-calculator](https://github.com/tomohxx/shanten-number-calculator)
  - [麻雀 和了判定（役の判定） アルゴリズム](http://hp.vector.co.jp/authors/VA046927/mjscore/mjalgorism.html)
  - [麻雀の数学](http://www10.plala.or.jp/rascalhp/mjmath.htm)
- Tenhou/mjlog
  - [mjlog形式について](http://m77.hatenablog.com/entry/2017/05/21/214529)
  - [NegativeMjark/tenhou-log](https://github.com/NegativeMjark/tenhou-log)
  - [mthrok/tenhou-log-utils](https://github.com/mthrok/tenhou-log-utils) mjlongのコンソールでの表示などができる
  - [天鳳牌譜(.mjlog形式)をXMLに直す](http://rausumaru.hatenablog.com/entry/2019/08/30/021154)
- Play at tenhou
  - [MahjongRepository/tenhou-python-bot](https://github.com/MahjongRepository/tenhou-python-bot)
  
## Build

```
$ ./build.sh
```
で`build` ディレクトリにビルドされる。

```
$ ./build/test/mahjong_test
```
でテストが実行できる。

