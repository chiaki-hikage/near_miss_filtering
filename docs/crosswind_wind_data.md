# 横風の検討に使った風のデータと、セグメントとの突き合わせの記録

comma2k19 の横風の検討 (`docs/crosswind_civic_feasibility.md`) で使った、次の 3 つのファイルの出どころをまとめる。

- `data/comma2k19_civic/_selection/wind/sfo_1min_all.csv`: SFO の ASOS 1 分値
- `data/comma2k19_civic/_selection/wind/metar_all.csv`: SFO・OAK・HAF の METAR
- `data/comma2k19_civic/_selection/wind/all_segments_wind.csv`: comma2k19 の全セグメントと SFO の風の突き合わせ

これらを作った処理は、リポジトリにスクリプトとして残っていない。2026-11-11 に、作業したときのセッションの記録 (Claude Code の作業ログ) から取得のコマンドと処理のコードを探し出し、この文書に写した。**データは変更していない。取得の処理は作り直していない。**

## 0. 要点

| ファイル | どう作ったか | 再現できるか |
|---|---|---|
| `sfo_1min_all.csv` | 2026-11-05 05:10 UTC に IEM (Iowa Environmental Mesonet) の配布 API から `curl` で取得 (§1.1) | **確かめていない**。URL は分かっているが、取り直していない (今回は取得しない方針)。配布元のデータが後から直されていれば、同じものにならない可能性がある |
| `metar_all.csv` | 同じ時刻に、同じ配布元から取得 (§1.2) | 同上 |
| `all_segments_wind.csv` | 2026-11-05 05:11 UTC に、作業用ディレクトリでその場で実行した Python の処理 (§2、コードは付録 A・B) | **再現できた**。付録のコードを作業用ディレクトリで流し、既存のファイルとバイト単位で一致した (2026-11-11)。ただし入力の一部 (Chunk 1〜2 の `segments.csv`) は保存されておらず、`raw_data/Chunk_1`・`Chunk_2` から作り直す必要がある |

「再現できない」と書いていたのは、スクリプトがリポジトリに無いためだった (`docs/crosswind_civic_feasibility.md` §11)。処理そのものは残っていたので、付録に写した。リポジトリのスクリプトにするかどうかは、まだ決めていない (§4)。

## 1. 気象データ (IEM から取得)

### 1.1 `sfo_1min_all.csv` (SFO の ASOS 1 分値)

取得のコマンド (2026-11-05 05:10 UTC。作業用ディレクトリで実行し、2026-11-06 に `data/` へ写した):

```bash
curl -s -m 300 "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py?station=SFO&tz=UTC&year1=2018&month1=5&day1=1&year2=2018&month2=11&day2=21&vars=sknt&vars=drct&vars=gust_sknt&vars=gust_drct&sample=1min&what=download&delim=comma" -o sfo_1min_all.csv
```

| 項目 | 内容 (2026-11-11 にファイルを読んで確認) |
|---|---|
| 行数・列 | 238,574 行。`station`, `station_name`, `valid(UTC)`, `sknt`, `drct`, `gust_sknt`, `gust_drct` |
| 観測所 | SFO (SAN FRANCISCO INTL) だけ |
| 期間 | 2018-05-01 00:00 〜 2018-11-05 04:09 (UTC)。2018-11-21 まで頼んだが、11-05 04:09 で終わっている |
| 欠け | 期間内の 1 分のうち約 12% は行が無い。行があっても 142 行は値が `M` (欠測) |
| 単位 | `sknt`・`gust_sknt` は kt、`drct`・`gust_drct` は deg (処理では kt に 0.514444 を掛けて m/s にしている)。風速は 2 分平均、突風は 1 分内の最大 (`docs/crosswind_civic_feasibility.md` §1.1 の記載。この文書では配布元の説明を読み直していない) |
| sha256 | `37fbe7ebbc7046707c489a7672da671975af07eb087155c7a79abc9c323c5e33` |

その前 (2026-11-05 04:50 ごろ) に、期間を 2018-07-27〜11-12 にした `sfo_1min.csv` も取っていた。こちらは `data/` に写していない。

### 1.2 `metar_all.csv` (METAR)

```bash
curl -s -m 120 "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=SFO&station=HAF&station=OAK&data=sknt&data=drct&data=gust&year1=2018&month1=5&day1=1&year2=2018&month2=11&day2=21&tz=Etc/UTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=3" -o metar_all.csv
```

| 項目 | 内容 |
|---|---|
| 行数・列 | 13,982 行。`station`, `valid`, `sknt`, `drct`, `gust` |
| 観測所 | SFO 4,893 行、OAK 4,887 行、HAF 4,202 行。**突き合わせに使ったのは SFO だけ** |
| 期間 | 2018-05-01 00:53 〜 2018-11-20 23:56 (UTC)。ほぼ毎時 |
| 欠け | `M` が欠測。`gust` は 93% が `M` (突風の報告があるときだけ値がある) |
| `report_type=3` | 配布元のパラメータ。意味は確かめていない |
| sha256 | `9163f49e0138ade3265f4e00b67b17e405841d09d4a53ac1ef4640aebf5d3a44` |

### 1.3 再現できない理由・確かめていないこと

- 取り直していないので、今の配布元から同じファイルが得られるかは分からない (配布元のデータの修正、API の変更がありうる)
- 取得の日時と URL は上のとおりで、ファイルの sha256 を記録した。取り直したときは sha256 で同じかどうかを確かめられる
- 列の定義 (2 分平均など) を配布元の説明で確かめ直していない

## 2. `all_segments_wind.csv` (セグメントと風の突き合わせ)

### 2.1 入力

| 入力 | 中身 | 出どころ | 保存 |
|---|---|---|---|
| `remote_chunk_3〜10.csv` | Chunk 3〜10 (Civic、1,653 セグメント) の先頭・中央・末尾の時刻と位置。時刻は GPS 時刻から 18 s を引いた UTC | `scripts/crosswind/civic_selection/collect.py` が Hugging Face の `Chunk_N.zip` を範囲指定で読んで作った | `data/comma2k19_civic/_selection/wind/` にある |
| `segments.csv` | Chunk 1〜2 (RAV4、382 セグメント) の先頭・中央・末尾の時刻と位置。時刻は GPS 時刻のまま (処理の中で 18 s を引く) | 手元の `raw_data/Chunk_1`・`Chunk_2` の `global_pose` から作った (付録 A) | **保存されていない**。付録 A で作り直せる |
| `sfo_1min_all.csv`, `metar_all.csv` | §1 | IEM | `data/` にある |

### 2.2 処理の内容 (付録 B のコードから読み取ったもの)

1. 各セグメントの中央の時刻 `(t0 + t1) / 2` を分に切り捨てて `tm` (UTC) とする
2. 進行方向 `hdg` は、先頭の位置から中央の位置への方位 (北 0°、時計回り)
3. `tm` と同じ分の SFO の 1 分値を付ける (`wind`, `gust1`, `drct`)
4. SFO の METAR のうち `tm` に最も近いものを、**前後 45 分以内**で付ける (`valid`, `m_sknt`, `m_drct`, `m_gustkt`)
5. 1 分値があれば 1 分値を、無ければ METAR を使う (`src` = `1min` / `metar` / `none`)。結果は `src` が `1min` 1,600 行、`metar` 435 行、`none` 0 行
6. `W` = 風速 [m/s]、`D` = 風向 [deg]、`G` = 突風 [m/s] (1 分値なら 1 分内の最大、METAR なら突風の報告。報告が無ければ欠測で、平均風速では埋めない)
7. 横風成分 `cross` = `W·|sin(D − hdg)|`、`crossG` = `G·|sin(D − hdg)|`
8. `local` = `tm` を America/Los_Angeles に直したもの

### 2.3 列

| 列 | 内容 |
|---|---|
| `chunk`, `drive`, `seg` | チャンク番号 (1〜10)、ドライブ ID、セグメント番号 |
| `t0`, `t1` | セグメントの最初と最後のフレームの UTC [Unix 秒] |
| `lat0`, `lon0`, `lat`, `lon` | 先頭と中央の位置 [deg] |
| `hdg` | 先頭→中央の方位 [deg] |
| `tm` | 中央の時刻 (分に切り捨て、UTC) |
| `wind`, `gust1`, `drct` | SFO の 1 分値 (m/s、m/s、deg)。無ければ空 |
| `valid`, `m_sknt`, `m_drct`, `m_gustkt` | 付けた METAR の時刻と値 (kt、deg、kt)。無ければ空 |
| `src` | 風の値の出どころ (`1min` / `metar`) |
| `W`, `D`, `G` | 使う風速・風向・突風 (m/s、deg、m/s) |
| `cross`, `crossG` | 横風成分 (m/s) |
| `local` | 現地時刻 |

2,035 行 (Chunk 1〜10 の全セグメント、178 ドライブ)。sha256 は `610b05b101e9e9bd7b4f184fb6ab6015f458f270c6c64fce3ff50441175209fd`。

### 2.4 2026-11-11 の再現の確認

1. 付録 A を作業用ディレクトリで流し、`raw_data/Chunk_1`・`Chunk_2` から `segments.csv` (382 行) を作った
2. `data/` の `sfo_1min_all.csv`・`metar_all.csv`・`remote_chunk_*.csv` の写しと一緒に、付録 B を流した
3. 出力の `all_segments_wind.csv` は、`data/comma2k19_civic/_selection/wind/all_segments_wind.csv` とバイト単位で一致した

付録のコードは、セッションの記録から**そのまま**写した (付録 A は出力先のパスだけを変え、後ろの表示の部分を削った)。リポジトリのスクリプトにはしていない。

## 3. 文書との食い違い

- `docs/crosswind_civic_feasibility.md` §1.1 は、1 分値が無いときの METAR を「前後 60 分以内で最も近いもの」としている。`all_segments_wind.csv` を作った処理 (付録 B) は**前後 45 分以内**だった。のちの `scripts/crosswind/xwind_search/join_wind.py` は 60 分 (3600 s) を使っている。`all_segments_wind.csv` では `src` が `none` の行が無いので、この違いで結果が変わったかは確かめていない
- `docs/crosswind_civic_feasibility.md` §1.1 の「1 分値の欠け 全体の約 25%」と、この文書の 1.1 の「約 12%」は、数えた対象が違う可能性がある。後者は 2018-05-01〜11-05 04:09 の 1 分のうち行が無いものの割合。前者の数え方は確かめていない

## 4. 今後の扱い (未決)

- 付録 A・B を `scripts/crosswind/civic_selection/` のスクリプトにするか。するなら、`segments.csv` の保存先と、45 分・60 分のどちらにそろえるかを決める必要がある
- 気象データを取り直して sha256 を比べるか

## 付録 A: `segments.csv` (Chunk 1〜2) を作った処理

セッションの記録の 2026-11-05 04:49 UTC の実行。リポジトリの直下で実行する。

```python
import sys; sys.path.insert(0,'scripts/vibration')
import _bootstrap, numpy as np, pandas as pd
from near_miss.io.comma2k19 import find_segments
import find_roll_bumps as F
OUT='<作業用ディレクトリ>'
rows=[]
for root in ('raw_data/Chunk_1','raw_data/Chunk_2'):
    for r in find_segments(root):
        g=np.load(r.path/'global_pose/frame_gps_times'); p=np.load(r.path/'global_pose/frame_positions'); v=np.load(r.path/'global_pose/frame_velocities')
        unix=315964800.0+g[:,0]*604800+g[:,1]
        lat,lon=zip(*[F.ecef_to_llh(*p[i]) for i in (0,len(p)//2,len(p)-1)])
        rows.append(dict(chunk=root[-7:],drive=r.drive_id,seg=r.index,t0=unix[0],t1=unix[-1],lat=lat[1],lon=lon[1],
                         lat0=lat[0],lon0=lon[0],speed=float(np.median(np.linalg.norm(v,axis=1)))))
d=pd.DataFrame(rows); d.to_csv(OUT+'/segments.csv',index=False)
```

## 付録 B: `all_segments_wind.csv` を作った処理

セッションの記録の 2026-11-05 05:11 UTC の実行。`segments.csv`・`remote_chunk_*.csv`・`sfo_1min_all.csv`・`metar_all.csv` のある作業用ディレクトリで実行する。最後の行が出力。途中の `print` は集計の表示だけで、出力には関係しない。

```python
import pandas as pd, numpy as np, glob
KT=0.514444
r=pd.concat([pd.read_csv(f) for f in sorted(glob.glob('remote_chunk_*.csv'))])
l=pd.read_csv('segments.csv'); l['t0']-=18; l['t1']-=18; l['chunk']=l.chunk.str[-1].astype(int)
cols=['chunk','drive','seg','t0','t1','lat0','lon0','lat','lon']
d=pd.concat([l[cols], r[cols]],ignore_index=True)
d['hdg']=np.degrees(np.arctan2((d.lon-d.lon0)*np.cos(np.radians(d.lat)),(d.lat-d.lat0)))%360
d['tm']=pd.to_datetime((d.t0+d.t1)/2,unit='s').dt.floor('min')
s=pd.read_csv('sfo_1min_all.csv'); s['t']=pd.to_datetime(s['valid(UTC)'])
for c in ('sknt','drct','gust_sknt'): s[c]=pd.to_numeric(s[c],errors='coerce')
s=s.drop_duplicates('t').set_index('t').sort_index()
s=s.assign(wind=s.sknt*KT, gust1=s.gust_sknt*KT)[['wind','gust1','drct']]
j=d.join(s,on='tm')
m=pd.read_csv('metar_all.csv',na_values='M'); m['valid']=pd.to_datetime(m.valid); m=m[m.station=='SFO'].sort_values('valid')
m=m.rename(columns={'sknt':'m_sknt','drct':'m_drct','gust':'m_gustkt'})[['valid','m_sknt','m_drct','m_gustkt']]
j=pd.merge_asof(j.sort_values('tm'),m,left_on='tm',right_on='valid',direction='nearest',tolerance=pd.Timedelta('45min'))
j['src']=np.where(j.wind.notna(),'1min',np.where(j.m_sknt.notna(),'metar','none'))
j['W']=j.wind.fillna(j.m_sknt*KT); j['D']=j.drct.fillna(j.m_drct)
j['G']=np.where(j.src=='1min', j.gust1, j.m_gustkt*KT)
j['cross']=j.W*np.abs(np.sin(np.radians(j.D-j.hdg))); j['crossG']=j.G*np.abs(np.sin(np.radians(j.D-j.hdg)))
j['local']=pd.to_datetime(j.tm).dt.tz_localize('UTC').dt.tz_convert('America/Los_Angeles')
# (ここに集計の表示が約 15 行あった。出力には関係しないので省いた)
j.to_csv('all_segments_wind.csv',index=False)
```

2.4 の確認は省く前の全文で行い、この付録の形 (表示を省いたもの) でも同じ出力になることを確かめた (2026-11-11)。
