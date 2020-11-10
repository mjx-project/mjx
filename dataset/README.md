# TrainDataGenerator

## 使い方
```shell script
mkdir -p <RESOURCE_PATH>/{json,disard,chi,pon,kan_closed,kan_opened,kan_added,riichi}
// <RESOURCE_PATH>/json にmjlogを変換したjsonファイルを置く.
cd cmake-build-debug/dataset
./generate_dataset DISCARD <RESOURCE_PATH>/json <RESOURCE_PATH>/discard
./generate_dataset CHI <RESOURCE_PATH>/json <RESOURCE_PATH>/chi
./generate_dataset PON <RESOURCE_PATH>/json <RESOURCE_PATH>/pon
./generate_dataset KAN_CLOSED <RESOURCE_PATH>/json <RESOURCE_PATH>/kan_closed
./generate_dataset KAN_OPENED <RESOURCE_PATH>/json <RESOURCE_PATH>/kan_opened
./generate_dataset KAN_ADDED <RESOURCE_PATH>/json <RESOURCE_PATH>/kan_added
./generate_dataset RIICHI <RESOURCE_PATH>/json <RESOURCE_PATH>/riichi
```


## データ形式
- Discard
```shell script
<Observation> <Event>
```
ここでのObservationには`possible_action`がsetされていない.

- Chi
```shell script
<Observation> <Event>
```
ここでのObservationの`possible_action`には1つ以上のChiが含まれる.

- Pon
```shell script
<Observation> <Event>
```
ここでのObservationの`possible_action`には1つ以上のPonが含まれる.

- KanClosed
```shell script
<Observation> {0 or 1}
```
ここでのObservationの`possible_action`にはKanClosedが含まれる.
その槓をしたかどうかが{0 or 1}で表される.

- KanOpened
```shell script
<Observation> {0 or 1}
```
ここでのObservationの`possible_action`にはKanOpenedが含まれる.
その槓をしたかどうかが{0 or 1}で表される.

- KanAdded
```shell script
<Observation> {0 or 1}
```
ここでのObservationの`possible_action`にはKanAddedが含まれる.
その槓をしたかどうかが{0 or 1}で表される.

- Riichi
```shell script
<Observation> {0 or 1}
```
ここでのObservationの`possible_action`にはRiichiが含まれる.
立直をしたかどうかが{0 or 1}で表される.
