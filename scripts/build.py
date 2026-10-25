#!/usr/bin/env python3
"""Build a self-contained EPUB and combined Markdown; never builds a website."""
from pathlib import Path
import argparse
import hashlib
import html
import io
import json
import os
import re
import zipfile
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import markdown
from PIL import Image
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build'
CSS = '''body{font-family:serif;line-height:1.75;color:#193746;margin:0 5%;}
h1{font-size:1.8em;line-height:1.4;color:#12384a;page-break-before:always;}
h2{font-size:1.28em;line-height:1.45;margin-top:1.8em;color:#167a85;}
h3{font-size:1.1em;} p{margin:.8em 0;} img{max-width:100%;height:auto;}
figure{margin:1.3em 0;} .caption{font-size:.86em;color:#52646c;line-height:1.55;}
pre{white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere;background:#eef2f1;padding:.8em;font-size:.76em;line-height:1.5;}
code{font-family:monospace;font-size:.88em;}pre code{font-size:1em;}
table{border-collapse:collapse;width:100%;font-size:.83em;line-height:1.5;}
th,td{border:1px solid #c8d4d3;padding:.45em;text-align:left;vertical-align:top;}
th{background:#e5efed;}a{color:#167a85;overflow-wrap:anywhere;}
blockquote{border-left:3px solid #cb7228;margin-left:0;padding-left:1em;}
.math-inline{vertical-align:middle;max-height:2.2em;width:auto;max-width:100%;}
.math-display{display:block;margin:1em auto;max-width:100%;width:auto;max-height:6em;}
'''


def documents(book):
    return ([{'id':'guide','title':'导读：从一次训练走向整个系统','source':'appendices/reading-guide.md'}]
            + book['chapters'] + book.get('appendices', []))


def validate(book):
    errors = []
    seen = set()
    for doc in documents(book):
        if doc['id'] in seen: errors.append('duplicate id '+doc['id'])
        seen.add(doc['id'])
        p = ROOT / doc['source']
        if not p.is_file():
            errors.append('missing '+doc['source']); continue
        text = p.read_text()
        if any(ord(c)<32 and c not in '\n\r\t' for c in text): errors.append('control character '+p.name)
        if re.search(r'TODO|待补正文|占位图|待生成图片',text): errors.append('placeholder '+p.name)
        if text.count('```')%2: errors.append('unbalanced code fence '+p.name)
        for ref in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text):
            if not (p.parent/ref).is_file(): errors.append('missing image '+ref+' in '+p.name)
        if doc['id'].startswith('ch') and not re.search(r'!\[',text): errors.append('chapter has no figure '+p.name)
    for p in (ROOT/'assets/images').glob('*.png'):
        try:
            with Image.open(p) as im: im.verify()
        except Exception as e: errors.append('invalid image '+p.name+': '+str(e))
    if errors: raise RuntimeError('\n'.join(errors))
    return {'chapters':len(book['chapters']),'appendices':len(book.get('appendices',[])),
            'chinese_characters':sum(len(re.findall(r'[\u4e00-\u9fff]',(ROOT/d['source']).read_text())) for d in documents(book)),
            'teaching_images':len(list((ROOT/'assets/images').glob('*.png')))}


def build():
    book = json.loads((ROOT/'book.json').read_text())
    summary = validate(book)
    BUILD.mkdir(exist_ok=True)
    resources = {}
    math_cache = {}
    ns = 'http://www.w3.org/1999/xhtml'

    def formula(match):
        raw = match.group(0)
        display = raw.startswith('$$')
        tex = raw[2:-2] if display else raw[1:-1]
        tex = re.sub(r'\s+',' ',tex).strip()
        if tex not in math_cache:
            # Standard LaTeX permits unbraced fractions; mathtext requires braces.
            compatible = tex.replace(r'\frac1D',r'\frac{1}{D}')
            compatible = compatible.replace(r'\mathsf T',r'\mathsf{T}')
            key = hashlib.sha256(tex.encode()).hexdigest()[:16]
            path = f'assets/math/{key}.png'
            stream = io.BytesIO()
            math_to_image('$'+compatible+'$',stream,prop=FontProperties(size=14),dpi=180,format='png',color='#193746')
            resources[path] = ('image/png',stream.getvalue())
            math_cache[tex] = path
        cls = 'math-display' if display else 'math-inline'
        return '<img class="'+cls+'" src="'+math_cache[tex]+'" alt="'+html.escape(tex,quote=True)+'" />'

    def math_replace(text):
        parts = re.split(r'(```.*?```|`[^`\n]+`)',text,flags=re.S)
        for i in range(0,len(parts),2):
            parts[i] = re.sub(r'\$\$.*?\$\$|(?<!\$)\$[^\n$]+\$(?!\$)',formula,parts[i],flags=re.S)
        return ''.join(parts)

    def image_replace(match,source):
        alt,ref = match.groups()
        p = (source.parent/ref).resolve()
        rel = p.relative_to(ROOT).as_posix()
        # Keep exact diagrams lossless, reduce large generated illustrations for reading.
        if p.name.startswith('fig-'):
            target = rel; mime='image/png'; data=p.read_bytes()
        else:
            target = str(Path(rel).with_suffix('.jpg')); mime='image/jpeg'
            stream = io.BytesIO()
            with Image.open(p) as im:
                im.convert('RGB').save(stream,format='JPEG',quality=92,optimize=True)
            data=stream.getvalue()
        resources[target] = (mime,data)
        return '!['+alt+']('+target+')'

    def linkify(root):
        # Make chapter-local reference URLs tappable while preserving code and existing links.
        pat = re.compile(r'https?://[^\s<>]+')
        for node in list(root.iter()):
            if not node.text or node.tag in {'a','code','pre','img'}: continue
            text = node.text; matches=list(pat.finditer(text))
            if not matches: continue
            node.text=text[:matches[0].start()]
            for i,m in enumerate(matches):
                value=m.group(0).rstrip('。；，）')
                suffix=m.group(0)[len(value):]
                a=ET.Element('a',href=value);a.text=value
                end=matches[i+1].start() if i+1<len(matches) else len(text)
                a.tail=suffix+text[m.end():end]
                node.insert(i,a)

    def xhtml(title,body):
        frag=ET.fromstring('<div>'+body+'</div>')
        linkify(frag)
        serialized=''.join(ET.tostring(e,encoding='unicode') for e in frag)
        return ('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
                f'<html xmlns="{ns}" xml:lang="zh-CN"><head><title>{html.escape(title)}</title>'
                '<link rel="stylesheet" type="text/css" href="book.css" /></head><body>'+serialized+'</body></html>')

    docs=documents(book)
    pages=[]
    combined=['# '+book['title']+'\n\n'+book['subtitle']+'\n\n2026年10月版\n']
    for d in docs:
        p=ROOT/d['source']; text=p.read_text()
        combined.append(text.replace('../assets/','../assets/'))
        text=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',lambda m:image_replace(m,p),text)
        text=math_replace(text)
        body=markdown.markdown(text,extensions=['tables','fenced_code','sane_lists'],output_format='xhtml')
        pages.append((d['id']+'.xhtml',d['title'],xhtml(d['title'],body)))
    (BUILD/'GPU-Modern-Computing.md').write_text('\n\n---\n\n'.join(combined))
    resources['book.css']=('text/css',CSS.encode())
    coverpath=ROOT/book['cover']; stream=io.BytesIO()
    with Image.open(coverpath) as im: im.convert('RGB').save(stream,format='JPEG',quality=94)
    resources['assets/images/cover.jpg']=('image/jpeg',stream.getvalue())
    cover=xhtml(book['title'],'<div><img src="assets/images/cover.jpg" alt="GPU 与现代 AI 计算封面" /></div>')
    pages.insert(0,('cover.xhtml','封面',cover))
    navitems=''.join(f'<li><a href="{name}">{html.escape(title)}</a></li>' for name,title,_ in pages)
    nav=('<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" xml:lang="zh-CN"><head><title>目录</title><link rel="stylesheet" type="text/css" href="book.css" /></head><body><nav epub:type="toc" id="toc"><h1>目录</h1><ol>'+navitems+'</ol></nav></body></html>')
    ncx=('<?xml version="1.0" encoding="utf-8"?><ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1"><head><meta name="dtb:uid" content="urn:gpu-modern-atlas:2026-10" /></head><docTitle><text>GPU 与现代 AI 计算</text></docTitle><navMap>'+''.join(f'<navPoint id="n{i}" playOrder="{i}"><navLabel><text>{html.escape(title)}</text></navLabel><content src="{name}" /></navPoint>' for i,(name,title,_) in enumerate(pages,1))+'</navMap></ncx>')
    manifest=['<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav" />','<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml" />']
    for i,(name,_,_) in enumerate(pages): manifest.append(f'<item id="p{i}" href="{name}" media-type="application/xhtml+xml" />')
    for i,(path,(mime,_)) in enumerate(resources.items()):
        props=' properties="cover-image"' if path=='assets/images/cover.jpg' else ''
        manifest.append(f'<item id="r{i}" href="{path}" media-type="{mime}"{props} />')
    modified=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    opf=('<?xml version="1.0" encoding="utf-8"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="uid"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="uid">urn:gpu-modern-atlas:2026-10</dc:identifier><dc:title>GPU 与现代 AI 计算：从硬件、训练到算子优化</dc:title><dc:language>zh-CN</dc:language><dc:creator>图解知识书项目</dc:creator><meta property="dcterms:modified">'+modified+'</meta></metadata><manifest>'+''.join(manifest)+'</manifest><spine toc="ncx">'+''.join(f'<itemref idref="p{i}" />' for i in range(len(pages)))+'</spine></package>')
    final_path=BUILD/'GPU-Modern-Computing.epub'
    path=BUILD/'GPU-Modern-Computing.epub.tmp'
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('mimetype','application/epub+zip',compress_type=zipfile.ZIP_STORED)
        z.writestr('META-INF/container.xml','<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml" /></rootfiles></container>')
        z.writestr('OEBPS/content.opf',opf)
        z.writestr('OEBPS/nav.xhtml',nav)
        z.writestr('OEBPS/toc.ncx',ncx)
        for name,_,data in pages: z.writestr('OEBPS/'+name,data)
        for name,(_,data) in resources.items(): z.writestr('OEBPS/'+name,data)
    with zipfile.ZipFile(path) as z:
        names=set(z.namelist())
        for name in names:
            if name.endswith(('.xhtml','.xml','.opf','.ncx')):
                root=ET.fromstring(z.read(name))
                if name.endswith('.xhtml'):
                    for node in root.iter():
                        for attr in ['src','href']:
                            ref=node.get(attr)
                            if ref and not re.match(r'https?://|#',ref):
                                target='OEBPS/'+ref.split('#')[0]
                                if target not in names: raise RuntimeError('broken EPUB link '+name+' '+ref)
        if z.testzip(): raise RuntimeError('corrupt EPUB zip')
    with path.open('rb') as handle: os.fsync(handle.fileno())
    os.replace(path,final_path)
    path=final_path
    summary['epub_math_figures']=len(math_cache)
    summary['epub_size_bytes']=path.stat().st_size
    (BUILD/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check: print(json.dumps(validate(json.loads((ROOT/'book.json').read_text())),ensure_ascii=False))
    else: build()
