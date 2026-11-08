"""V6-5: copy frozen CSV strings; no training or network access."""
import csv,hashlib,json,math,datetime
from pathlib import Path
ROOT=Path('data/project')
OUT=ROOT/'results/v6_split_20261105'
SOURCES=[(ROOT/'results/v6_baselines_20261105/candidate_V6_3_B_restored.csv','81f063d8b43403e9d131bd8b9610ebb9144ab99e73f4e1104723eab2b37f9137'),(ROOT/'results/v6_baselines_20261105/candidate_V6_4_B_raw.csv','8627d6475ad33c1dc3db23f794a6af37ff3d1d1fd4a0ecb065b361ac98516274')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8',newline='') as f:
        reader=csv.DictReader(f);assert reader.fieldnames==['id','height','agb'];return list(reader)
assert not OUT.exists(),'Output directory exists; stop without overwrite'
for p,h in SOURCES:assert sha(p)==h,(str(p),'hash mismatch')
a,b=[rows(p) for p,h in SOURCES]
assert len(a)==len(b)==185130
assert all(x['id']==y['id'] for x,y in zip(a,b))
assert all(math.isfinite(float(x['height'])) and math.isfinite(float(y['agb'])) for x,y in zip(a,b))
OUT.mkdir()
path=OUT/'candidate_V6_5_h_restored_a_raw.csv'
with path.open('x',encoding='utf-8',newline='') as f:
    writer=csv.writer(f);writer.writerow(['id','height','agb'])
    writer.writerows((x['id'],x['height'],y['agb']) for x,y in zip(a,b))
actual=rows(path)
assert len(actual)==185130
assert all(z['id']==x['id']==y['id'] and z['height']==x['height'] and z['agb']==y['agb'] for z,x,y in zip(actual,a,b))
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'path':str(path),'sha256':sha(path),'rows':len(actual),'id_order_matches_both_sources':True,'height_strings_exact_V6_3':True,'agb_strings_exact_V6_4':True,'all_values_finite':True,'sources':[{'path':str(p),'sha256':h} for p,h in SOURCES],'uploaded':False}
(OUT/'generation_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))
