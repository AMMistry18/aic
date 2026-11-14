"""Audit citations, evidence-derived table macros, PDF text and rendered pages."""
import hashlib
import json
from pathlib import Path
import re
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'paper'


def main():
    tex=(PAPER/'main.tex').read_text();bib=(PAPER/'ref.bib').read_text()
    used={key.strip() for group in re.findall(r'\\cite\{([^}]+)\}',tex) for key in group.split(',')}
    defined=set(re.findall(r'@\w+\{([^,]+),',bib))
    if used-defined:raise RuntimeError('Undefined citations: '+str(used-defined))
    if defined-used:raise RuntimeError('Unused citations: '+str(defined-used))
    if '\\fbox' in tex:raise RuntimeError('Unresolved structural figure')
    data=json.loads((PAPER/'results/recovery_snapshot_20261112.json').read_text())
    for result in data['results']:
        if result['status']!='completed':continue
        original=ROOT/result['evidence_dir']/result['run']/'verified.json'
        current=json.loads(original.read_text())
        for key in ['verified_success','reason','world_sha256','terminal_metrics','high_rate_wrist_audit']:
            if result[key]!=current[key]:raise RuntimeError('Snapshot differs from verifier: '+result['run']+' '+key)
    for row in json.loads((PAPER/'figures/provenance.json').read_text()).values():
        path=ROOT/row['source']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Figure source changed')
    log=(PAPER/'main.log').read_text(errors='replace')
    problems=[line for line in log.splitlines() if 'Overfull \\hbox' in line or 'undefined' in line.lower() or 'LaTeX Error' in line]
    doc=pymupdf.open(PAPER/'main.pdf');qa=PAPER/'qa';qa.mkdir(exist_ok=True)
    pages=[]
    for i,page in enumerate(doc):
        page.get_pixmap(matrix=pymupdf.Matrix(1.4,1.4)).save(qa/f'page_{i+1}.png')
        text=page.get_text();pages.append(dict(page=i+1,words=len(text.split()),width_pt=page.rect.width,height_pt=page.rect.height))
    (qa/'extracted_text.txt').write_text('\n\n'.join(p.get_text() for p in doc),encoding='utf-8')
    report=dict(page_count=len(doc),pages=pages,citations=len(used),unresolved_citations=sorted(used-defined),
        compile_problems=problems,raw_evidence_compared=True,result_status='partial_development',
        pending_results_marked='Results pending' in tex,visual_inspection='Required after this rendering')
    (qa/'checks.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    if problems:raise RuntimeError('Compile problems remain')


if __name__=='__main__':main()
