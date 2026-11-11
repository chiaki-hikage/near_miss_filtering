"""Civic の取得セグメントを選ぶ (横風・bicycle model の当てはめ用)。"""
import sys, pandas as pd, numpy as np
S = sys.argv[1]; out = sys.argv[2]
d = pd.read_csv(f'{S}/all_segments_wind.csv'); d = d[d.chunk >= 3].copy()
r = pd.concat([pd.read_csv(f'{S}/remote_chunk_{n}.csv') for n in range(3, 11)])
d = d.merge(r[['drive', 'seg', 'lat1', 'lon1']], on=['drive', 'seg'], how='left')
R = 6371e3
def xy(lat, lon):
    return np.c_[R*np.radians(lat), R*np.radians(lon)*np.cos(np.radians(37.7))]
d['v_est'] = np.hypot(*(xy(d.lat1, d.lon1) - xy(d.lat0, d.lon0)).T) / 60
d['windy'] = d.G >= 12.5
d['calm'] = (d.W <= 3.1) & (d.G.isna() | (d.G <= 5.2))
wd = set(d[d.windy].drive)
reason = pd.Series('', index=d.index)
reason[d.drive.isin(wd)] = 'windy_drive'
# 強風セグメントの通った点 (始点・中点・終点) から 150 m 以内を通る弱風セグメント
P = np.vstack([xy(d.loc[d.windy, a], d.loc[d.windy, o]) for a, o in (('lat0','lon0'),('lat','lon'),('lat1','lon1'))])
def near(row):
    q = np.vstack([xy(np.array([row[a]]), np.array([row[o]])) for a, o in (('lat0','lon0'),('lat','lon'),('lat1','lon1'))])
    return (np.hypot(*(q[:, None, :] - P[None]).transpose(2, 0, 1)) < 150).any()
m = d.calm & ~d.drive.isin(wd) & (d.v_est > 5) & d.apply(near, axis=1)
cand = d[m].copy()
# ドライブごとに均して最大 150 件 (同じ通勤路の繰り返しが多いため)
cand['k'] = cand.groupby('drive').cumcount(); cand['r'] = np.random.default_rng(0).random(len(cand))
cand = cand.sort_values(['k', 'r']).head(150)
reason[cand.index] = 'calm_same_site'
# 当てはめ用: 弱風・走行中 (v_est > 10) のセグメントを日付に散らして足す
rest = d[d.calm & (reason == '') & (d.v_est > 10)].copy()
rest['date'] = rest.drive.str.slice(17, 27)

sel = d[reason != ''].assign(reason=reason[reason != ''])
sel = sel[['chunk', 'drive', 'seg', 'reason', 'W', 'G', 'D', 'cross', 'crossG', 'src', 'v_est', 'lat', 'lon', 'local']]
sel.sort_values(['chunk', 'drive', 'seg']).to_csv(out, index=False)
print(sel.reason.value_counts(), len(sel), sel.drive.nunique())
