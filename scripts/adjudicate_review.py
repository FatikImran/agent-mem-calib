"""Bind completed human review cards to the original screen and freeze eligible parents."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from scripts.screen_400 import sha
from scripts.verify_heldout_split import build
from src.data.loader import load_tasks

ROOT = Path('phase2/evidence')
REVIEW = Path('phase2/operator-review-screen-v1.md')
SCREEN = ROOT / 'screen-400-v1.json'


def save_frozen(path, value):
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError(f'Existing frozen artifact differs: {path}')
    else:
        with path.open('x', encoding='utf8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')


def main():
    rows=json.loads(SCREEN.read_bytes())
    by_id={r['task_id']:r for r in rows}
    tasks={t.task_id:t for t in load_tasks()}
    split=build()
    buckets=defaultdict(list)
    for r in rows:
        c=r.get('selected')
        if c:
            for cond in ('partial','unrelated'):
                if c[cond]:
                    buckets[cond,r['domain']].append((r['task_id'],cond))
    expected=[]
    for cond,quota in [('partial',20),('unrelated',40)]:
        groups=[sorted(g,key=lambda x:sha('review-v1/'+'/'.join(x))) for k,g in sorted(buckets.items()) if k[0]==cond]
        count=0
        while any(groups) and count<quota and len(expected)<40:
            for g in groups:
                if g and count<quota and len(expected)<40:
                    expected.append(g.pop(0)); count+=1
    cards=re.split(r'(?m)^## (?=\d+\.)',REVIEW.read_text(encoding='utf-8-sig'))[1:]
    reviews=[]
    for i,card in enumerate(cards):
        head=re.match(r'(\d+)\. (qa_\d+) · (\w+)',card)
        number,tid,condition=head.groups()
        assert int(number)==i+1 and (tid,condition)==expected[i]
        row=by_id[tid]; c=row['selected']
        for label,text in [('Request',row['request']),('Injection',c[condition])]:
            assert re.search('^'+label+': (.*)$',card,re.M)[1]==text
        field=re.search(r'^Decision:[ \t]*(.*)$',card,re.M)[1].strip()
        reason=card.split('Reason or correction:',1)[1].strip()
        decision=field.split()[0].rstrip(',').lower() if field else ''
        if tid=='qa_234' and not decision and reason.endswith('Accept with flag.'):
            decision='accept'
        if decision not in ('accept','reject'): raise ValueError((tid,field,reason))
        reviews.append(dict(number=i+1,task_id=tid,condition=condition,decision=decision,decision_field=field,reason=reason))
    assert len(reviews)==len(expected)==40
    rejected={(r['task_id'],r['condition']) for r in reviews if r['decision']=='reject'}
    ledger=[]
    for r in rows:
        tid=r['task_id']
        assert (tid in split['development'])==(r['split']=='development')
        item={'task_id':tid,'split':r['split'],'screen_status':r['status'],'reason':r['reason'],'eligible':False}
        if (tid,'unrelated') in rejected:
            item['reason']='review_rejected_required_unrelated_arm'
        elif 'selected' in r:
            item.update(eligible=True,annotation=r['selected'].copy())
            if (tid,'partial') in rejected:
                item['annotation']['partial']=None
        ledger.append(item)
    citations={}
    for tid,source in [('qa_327','toolace_5607'),('qa_305','toolace_702'),('qa_251','toolace_4092')]:
        turns=[t for t in tasks[tid].memory_context if t.source_id==source]
        citations[tid]={'source':source,'linked_turn_count':len(turns)}
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SCREEN,REVIEW,ROOT/'heldout-split-v1.json',Path('phase2/proposed-enrollment-and-scoring.md'))}
    save_frozen(ROOT/'human-review-adjudication-v1.json',{'hashes':hashes,'cards':reviews,'citation_checks':citations})
    save_frozen(ROOT/'paired-enrollment-v1.json',{'hashes':hashes,'parents':ledger})
    for side in ('development','heldout'):
        accepted=[r for r in ledger if r['eligible'] and r['split']==side]
        m=sum(bool(r['annotation']['partial']) for r in accepted)
        print(side,len(accepted),m,2*len(accepted)+m)
    print(Counter(r['decision'] for r in reviews),citations)
if __name__=='__main__': main()
