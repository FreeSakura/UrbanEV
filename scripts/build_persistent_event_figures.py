"""Build publication figures from committed summaries, without new experiments."""
from pathlib import Path
import argparse,csv,json
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor,white,black
from pdf2image import convert_from_path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/persistent_events/figures'
W=468
BLUE=HexColor('#24567B');GRAY=HexColor('#D6DEE3');ORANGE=HexColor('#C06428')

def text(c,x,y,s,size=10,bold=False,color=black):
    c.setFillColor(color);c.setFont('Helvetica-Bold' if bold else 'Helvetica',size);c.drawString(x,y,s)

def finish(c,name):
    c.save()
    pages=convert_from_path(str(OUT/f'{name}.pdf'),dpi=200)
    assert len(pages)==1
    pages[0].save(OUT/f'{name}.png')

def bits(c,x,y,bits,label):
    text(c,x,y+9,label,8.5)
    x+=55
    for i,b in enumerate(bits):
        c.setFillColor(BLUE if b=='1' else white);c.setStrokeColor(HexColor('#8A969E'))
        c.rect(x+i*18,y,16,19,stroke=1,fill=1)
        text(c,x+i*18+5,y+6,b,9,color=white if b=='1' else black)

def structure():
    name='structural_regime';c=canvas.Canvas(str(OUT/f'{name}.pdf'),pagesize=(W,222))
    for x,head,k,rows,lab in [
        (8,'Short window',4,['111000','000111','111111'],['100','001','111']),
        (240,'Long window',5,['1110000','0000111','1110111'],['100','001','101'])]:
        text(c,x,202,head,11,True);text(c,x,186,f'L = 3, K = {k}',9)
        for y,b,label in zip([146,112,78],rows,['Witness 1','Witness 2','Union']):bits(c,x,y,b,label)
        text(c,x,53,'Event labels: '+lab[0]+' and '+lab[1],9)
        text(c,x,36,'Union trace gives '+lab[2],10,True)
        text(c,x,16,'Desired 101 is impossible' if k==4 else 'Desired 101 is realizable',8.5,color=ORANGE if k==4 else BLUE)
    finish(c,name)

def flow(rows):
    ambiguous=[r for r in rows if int(r['ambiguous_labels'])>0]
    exact=[r for r in ambiguous if r['geometry_status']=='PAIRWISE_EXACT']
    inc=[r for r in ambiguous if r['geometry_status']=='PAIRWISE_INSUFFICIENT']
    gaps=sum(float(r['implication_width_gap'])>1e-10 for r in inc)
    decisions=sum(r['implication_direction']=='UNRESOLVED' and r['DP_direction']!='UNRESOLVED' for r in inc)
    data={'ambiguous_panels':len(ambiguous)//3,'exact_panels':len(exact)//3,'incomplete_panels':len(inc)//3,
          'exact_comparisons':len(exact),'incomplete_comparisons':len(inc),'endpoint_changes':gaps,'decision_changes':decisions}
    assert list(data.values())==[470,323,147,969,441,110,1]
    name='structure_to_decision';c=canvas.Canvas(str(OUT/f'{name}.pdf'),pagesize=(W,228))
    text(c,8,207,'470 panels with ambiguous labels',12,True)
    text(c,8,185,'Implication-exact: 323 panels / 969 comparisons',10,color=BLUE)
    text(c,8,167,'No additional endpoint or strict-direction change',9)
    text(c,8,140,'Implication-incomplete: 147 panels',10,True)
    x0=173;scale=254/441
    for y,label,count,color in [(109,'All comparisons',441,GRAY),(76,'Endpoint changes',110,BLUE),(43,'Strict-direction changes',1,ORANGE)]:
        text(c,8,y+5,label,9.5);c.setFillColor(color);c.rect(x0,y,count*scale,17,fill=1,stroke=0)
        text(c,x0+count*scale+6,y+4,str(count),10,True)
    text(c,8,12,'Nested counts within the incomplete branch; panels are not independent trials.',8)
    finish(c,name);return data

def intervals(rows):
    r=next(r for r in rows if r['phase']=='AP2' and r['series']=='Changping' and r['K']=='72'
           and r['scope']=='origin_week' and r['origin_start']=='22728' and r['model_A']=='constant' and r['model_B']=='hist_gradient_boosting')
    data={m:[float(r[m+'_lower']),float(r[m+'_upper'])] for m in ['independent','implication','cover_LP','cover_ILP','DP']}
    name='changping_intervals';c=canvas.Canvas(str(OUT/f'{name}.pdf'),pagesize=(W,211))
    text(c,8,190,'Changping 2015-10-04',11,True);text(c,8,174,'Constant minus gradient boosting; K = 72, L = 3',9)
    left=172;right=451;lo=-.05;hi=.005
    xx=lambda v:left+(v-lo)/(hi-lo)*(right-left)
    c.setStrokeColor(HexColor('#A7AFB5'));c.setLineWidth(.5)
    for v in [-.05,-.04,-.03,-.02,-.01,0]:
        c.line(xx(v),43,xx(v),151);text(c,xx(v)-12,30,f'{v:.2f}',8)
    c.setStrokeColor(black);c.setDash(2,2);c.line(xx(0),43,xx(0),152);c.setDash()
    for y,label,key,col in [(140,'Independent','independent',HexColor('#787F85')),(101,'Unary implications','implication',BLUE),(62,'Complete covers / trajectory','cover_LP',ORANGE)]:
        text(c,8,y-3,label,8.7)
        a,b=data[key];c.setStrokeColor(col);c.setLineWidth(2.4);c.line(xx(a),y,xx(b),y)
        c.setLineWidth(1.3);c.line(xx(a),y-5,xx(a),y+5);c.line(xx(b),y-5,xx(b),y+5)
    text(c,172,10,'Mean Brier-risk difference (feasible range)',8.5)
    finish(c,name);return data

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT,help='Directory for generated figures; use a temporary directory in CI')
    OUT=parser.parse_args().output
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'artifacts/summaries/persistent_event_complete_covers_20260917/comparison_results.csv'
    with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
    structure();data={'structure':flow(rows),'changping':intervals(rows)}
    (OUT/'figure_data.json').write_text(json.dumps(data,indent=2)+'\n')
    print('Built three vector PDFs, three PNGs and figure_data.json from frozen evidence')

if __name__=='__main__':main()
