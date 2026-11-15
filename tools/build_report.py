from pathlib import Path
import argparse,hashlib
import re,html
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
root=Path(__file__).resolve().parents[1]
from reportlab import rl_config
rl_config.invariant = 1
body=ParagraphStyle('Body',fontName='Helvetica',fontSize=9,leading=11.2,spaceAfter=5)
heading=ParagraphStyle('Heading',parent=body,fontName='Helvetica-Bold',fontSize=10.5,leading=12,spaceBefore=4,spaceAfter=5,keepWithNext=True)
title=ParagraphStyle('Title',parent=heading,fontSize=16,leading=18)
cell=ParagraphStyle('Cell',parent=body,fontSize=8.2,leading=9.7,spaceAfter=0)
def markup(s):
 s=html.escape(s).replace('→',' -&gt; ').replace('−','-').replace('≈','~').replace('×',' x ').replace('²','^2').replace('≥','&gt;=')
 s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
 s=re.sub(r'`([^`]+)`',r'<font name="Courier" size="7.8">\1</font>',s)
 return s
parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=root/'report.pdf');args=parser.parse_args()
story=[]; rows=[]
def flush():
 if not rows:return
 t=Table([[Paragraph(markup(x),cell) for x in r] for r in rows],colWidths=[106,417],repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9edf2')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.5,colors.grey),('BOTTOMPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4)]))
 story.extend([t,Spacer(1,5)]);rows.clear()
for line in (root/'report.md').read_text().splitlines():
 if line.startswith('|'):
  r=[x.strip() for x in line.strip('|').split('|')]
  if not all(set(x)<=set('-: ') for x in r):rows.append(r)
  continue
 flush()
 if line=='<!-- PAGEBREAK -->':story.append(PageBreak())
 elif line.startswith('# '):story.append(Paragraph(markup(line[2:]),title))
 elif line.startswith('## '):story.append(Paragraph(markup(line[3:]),heading))
 elif line.strip():story.append(Paragraph(markup(line),body))
flush()
def footer(c,d):
 c.setFont('Helvetica',8);c.drawString(36,20,'Model quantization | Evidence audit | 15 November 2026');c.drawRightString(A4[0]-36,20,str(d.page))
SimpleDocTemplate(str(args.out),pagesize=A4,leftMargin=36,rightMargin=36,topMargin=30,bottomMargin=34,title='Model quantization evidence audit',author='Experiment submission',subject='report.md SHA-256: '+hashlib.sha256((root/'report.md').read_bytes()).hexdigest()).build(story,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader
pdf=PdfReader(args.out);print('Pages',len(pdf.pages));print('Words',len((root/'report.md').read_text().split()));assert len(pdf.pages)<=2
print('Extracted characters',sum(len(p.extract_text()) for p in pdf.pages))
