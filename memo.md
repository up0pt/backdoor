## 実験メモ

### 2025/05/07
backdoorの印を，BadNetsに合わせて，人の字の形のように配置．
9c8947db4c2096ff4313393b10e41dd54f68331a

目的：
攻撃の成功確率が低いのは，印が弱いから？という仮説を検証するため．（防御なし）

考察：
まず、実装の変更として、CNNのシード値を固定しているので、初期からモデルの類似度が高い。その結果、ほとんど同一（コサイン類似度0.75-0.99）のモデルとなる。
精度を見ると、clean精度も攻撃精度も不安定かつ低精度である。
![alt text](results/20250507_105230/metrics.png)

示唆：
攻撃側からするとランダム初期値がバラバラの方が良い？

TODO:
- CNN初期値の指定をなくして実験
    - その結果，clean acc が安定し，1程度に収束．
    ![alt text](results/20250508_014442/metrics.png)
    - もし攻撃がなかったらどうか？
    /home/members/nakadam/backdoor/results/20250508_052754
    ![alt text](results/20250508_055212/metrics.png)
- Syros＋の実験条件の確認


---
### 2025/05/08
backdoorの印をよりでかくした．(job 2272)
```
clients=30
attackers=16
selection=random
rounds=40
pdr=0.7
boost=5
Device:cuda
PDR:0.7
Boost:5.0
ClipG:0.0001
ClipL:1.0
```
とりあえずattackerを16にしてみた（防御手法なしで）results/20250512_010727
![alt text](results/20250512_010727/metrics.png)
→防御してないのに、収束してる（seedの固定による？）
seed 固定していない防御なし・攻撃ありで確かめる。
![alt text](results/20250512_075226/metrics.png)
やっぱり収束してる。。。

---
### 2025/05/12

Sentinel 実装！！

Sentinel実装後、原点回帰の攻撃(job 2290)
![alt text](results/20250512_102244/metrics.png)
なぜか2*2の模様が強い。
job 2294で40roundやってみる。

Sentinelの効果をみるために、sentinel動作(job 2293)
![alt text](results/20250512_120246/metrics.png)
なぜかSentinel入れても防御できていない...
(31ebfeb91594dd2d272437cfe955254523a03ecb)
Sentinel更新後（履歴のクライアントごと化）でもだめ
![](results/20250512_142904/metrics.png)

とりあえず、閾値を極端にしてクライアント自分自身のみでattackされないことを確認したい。


#### 別軸でselectionをやっている。PageRankではどうなるか？
job 2299





