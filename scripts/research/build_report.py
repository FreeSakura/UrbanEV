"""Build the Chinese V3 report using XeLaTeX (ctex required)."""
import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
def esc(value):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}', '~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(mapping.get(c,c) for c in value).replace('−',r'\textminus{}').replace('–','--').replace('—','---')

def inline(value):
    saved=[]
    def hold(tex):
        token=f'ZZTOKEN{len(saved)}ZZ';saved.append(tex);return token
    value=re.sub(r'\$([^$]+)\$',lambda m:hold('$'+m.group(1)+'$'),value)
    value=re.sub(r'`([^`]+)`',lambda m:hold(r'\allowbreak{}\texttt{'+esc(m.group(1)).replace('/', r'/\allowbreak{}').replace(r'\_', r'\_\allowbreak{}').replace('-', r'-\allowbreak{}')+r'}\allowbreak{}'),value)
    value=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:hold(r'\href{'+m.group(2).replace('%',r'\%')+'}{'+esc(m.group(1))+'}'),value)
    value=re.sub(r'\*\*([^*]+)\*\*',lambda m:hold(r'\textbf{'+esc(m.group(1))+'}'),value)
    value=esc(value)
    # Tokens may be nested in bold markup; repeat until fully expanded.
    for _ in range(3):
        for i,tex in enumerate(saved):value=value.replace(f'ZZTOKEN{i}ZZ',tex)
    if 'ZZTOKEN' in value:raise ValueError('Unexpanded markup token')
    return value

def table(lines):
    rows=[[c.strip() for c in line.strip().strip('|').split('|')] for line in lines]
    rows=[r for r in rows if not all(re.fullmatch(r':?-+:?',c) for c in r)]
    n=len(rows[0]);assert all(len(r)==n for r in rows)
    widths={2:[.30,.65],3:[.21,.35,.36],4:[.15,.29,.27,.20]}.get(n,[.90/n]*n)
    spec='@{}'+''.join(r'>{\raggedright\arraybackslash}p{'+str(w)+r'\linewidth}' for w in widths)+'@{}'
    row=lambda r:' & '.join(inline(c) for c in r)+r' \\'
    header=row([r'**'+c+'**' for c in rows[0]])
    out=[r'\par\medskip\noindent\begin{minipage}{\linewidth}\small\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.2}',
         r'\begin{tabular}{'+spec+'}',r'\toprule',header,r'\midrule']
    out += [row(r) for r in rows[1:]]
    return '\n'.join(out+[r'\bottomrule\end{tabular}\end{minipage}\par\medskip'])

def convert(text):
    lines=text.splitlines();out=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('# '):i+=1;continue
        if line.startswith('$$'):
            block=[line[2:]];i+=1
            while not block[-1].rstrip().endswith('$$'):
                if i>=len(lines):raise ValueError('Unclosed display math')
                block.append(lines[i]);i+=1
            block[-1]=block[-1].rstrip()[:-2]
            math='\n'.join(block).strip()
            out.append(r'\begin{equation}'+ '\n'+math+'\n'+r'\end{equation}');continue
        if line.startswith('|'):
            block=[]
            while i<len(lines) and lines[i].strip().startswith('|'):block.append(lines[i]);i+=1
            out.append(table(block));continue
        if line.startswith('## '):
            title=line[3:]
            if title.startswith('附录'):
                letter=re.match(r'附录([A-Z])',title).group(1)
                pagebreak=r'\clearpage' if letter in ('A','C') else ''
                out.append(pagebreak+r'\setcounter{subsection}{0}\renewcommand{\thesubsection}{'+letter+r'.\arabic{subsection}}\section*{'+inline(title)+'}'+r'\addcontentsline{toc}{section}{'+inline(title)+'}')
            else:
                title=re.sub(r'^\d+\.\s*','',title)
                out.append(r'\section{'+inline(title)+'}')
            i+=1;continue
        if line.startswith('### '):
            title=re.sub(r'^(?:\d+\.\d+|[A-Z]\.\d+)\s*','',line[4:])
            out.append(r'\subsection{'+inline(title)+'}')
            i+=1;continue
        if line.startswith('- ') or re.match(r'^\d+\. ',line):
            ordered=not line.startswith('- ');env='enumerate' if ordered else 'itemize';items=[]
            pattern=r'^\d+\. ' if ordered else r'^- '
            while i<len(lines) and re.match(pattern,lines[i].strip()):
                items.append(r'\item '+inline(re.sub(pattern,'',lines[i].strip())));i+=1
            out.append(r'\begin{'+env+'}\n'+'\n'.join(items)+'\n'+r'\end{'+env+'}');continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\$\$|\||- |\d+\. )',lines[i].strip()):
            paragraph.append(lines[i].strip());i+=1
        out.append(inline(' '.join(paragraph))+'\n')
    return '\n\n'.join(out)


def github_links(text, source):
    """Resolve each Markdown source before combining relocated documents."""
    def link(match):
        label, href = match.groups()
        parts = urlsplit(href)
        if not parts.scheme and not parts.netloc:
            target = (source.parent / parts.path).resolve() if parts.path else source.resolve()
            relative = target.relative_to(ROOT).as_posix()
            href = 'https://github.com/FreeSakura/UrbanEV/blob/main/' + relative
            if parts.query:
                href += '?' + parts.query
            if parts.fragment:
                href += '#' + parts.fragment
        return '[' + label + '](' + href + ')'
    return re.sub(r'\[([^\]]+)\]\(([^)]+)\)', link, text)


def main():
    directory = ROOT / 'paper/research'
    directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'docs/reports/audit/PAIRED_AUDIT_V3_REPORT.md'
    derivation = ROOT / 'docs/theory/DERIVATION_V3.md'
    body = github_links(source.read_text(encoding='utf-8'), source)
    body += '\n\n## 附录A. 完整证明\n\n'
    body += github_links(derivation.read_text(encoding='utf-8'), derivation)
    fontset = 'windows' if os.name == 'nt' else 'fandol'
    preamble = r"""\documentclass[UTF8,11pt,a4paper,fontset=FONTSET]{ctexart}
\usepackage{amsmath,amssymb,mathtools,booktabs,longtable,array,enumitem,xcolor,hyperref,fancyhdr,textcomp}
\usepackage[margin=24mm,headheight=15pt]{geometry}
\hypersetup{unicode=true,colorlinks=true,urlcolor=blue,linkcolor=blue,pdftitle={UrbanEV Theory and Experiment Feedback V3},pdfauthor={FreeSakura}}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{0.3em}
\allowdisplaybreaks
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{UrbanEV 理论与实验反馈 V3}\fancyfoot[C]{\thepage}
\begin{document}
\begin{titlepage}\centering\vspace*{30mm}
{\Huge UrbanEV 理论深化\par}\vspace{8mm}
{\LARGE 与实验反馈 V3\par}\vspace{20mm}
{\Large 共享缺失观测下的配对风险界\par}\vspace{7mm}
{\large 精确识别、单点核验与真实缺失验证\par}
\vfill FreeSakura\par 2026年9月9日\par
\vspace{10mm}{\small 研究工作报告：数学结论与开发证据分开陈述}
\end{titlepage}
\tableofcontents\clearpage
""".replace('FONTSET', fontset)
    tex = directory / 'UrbanEV_Theory_Feedback_V3.tex'
    tex.write_text(preamble+convert(body)+'\n\\end{document}\n', encoding='utf-8', newline='\n')
    executable = shutil.which('xelatex')
    if not executable:
        raise SystemExit('XeLaTeX and ctex are required')
    for _ in range(2):
        subprocess.run([executable, '-interaction=nonstopmode', '-halt-on-error', tex.name], cwd=directory, check=True)
    log = tex.with_suffix('.log').read_text(encoding='utf-8', errors='replace')
    if any(x in log for x in ('Overfull \\hbox', 'Overfull \\vbox', 'Missing character:', 'undefined references')):
        raise SystemExit('Research PDF layout/reference gate failed')
    final = directory / 'UrbanEV_Theory_Feedback_V3.pdf'
    print(final.name)

if __name__ == '__main__':
    main()
