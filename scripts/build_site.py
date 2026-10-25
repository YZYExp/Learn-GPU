#!/usr/bin/env python3
"""Build the reading website with only Markdown, Pillow and latex2mathml."""
import argparse
import html
import json
import os
import posixpath
import re
import shutil
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
import xml.etree.ElementTree as ET

import markdown
from PIL import Image
from latex2mathml.converter import convert

ROOT=Path(__file__).resolve().parents[1]
REPO='https://github.com/YZYExp/Learn-GPU'


def run():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-path',default='/Learn-GPU/')
    ap.add_argument('--output',default='build/site')
    ap.add_argument('--epub',help='An explicitly supplied, validated reading edition')
    args=ap.parse_args()
    base='/'+args.base_path.strip('/')+'/' if args.base_path.strip('/') else '/'
    out=ROOT/args.output
    if out.resolve()==ROOT or ROOT not in out.resolve().parents: raise ValueError('output must be a project subdirectory')
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    book=json.loads((ROOT/'book.json').read_text())
    guide={'id':'guide','title':'导读：从一次训练走向整个系统','source':'appendices/reading-guide.md','slug':'guide'}
    docs=[guide]+book['chapters']+[dict(a,slug=a['id']) for a in book['appendices']]
    slugs=[d['slug'] for d in docs]
    if len(slugs)!=len(set(slugs)): raise ValueError('duplicate slug')
    assets={}
    def url(path=''): return base+path.lstrip('/')
    def asset(ref,source):
        p=(source.parent/ref).resolve()
        if not p.is_file() or ROOT not in p.parents: raise ValueError('invalid image '+ref)
        name=p.name
        # Lossless precision diagrams; optimized JPEG illustrations, retaining proportions.
        target='assets/images/'+(name if name.startswith('fig-') else p.stem+'.jpg')
        if target not in assets:
            dest=out/target;dest.parent.mkdir(parents=True,exist_ok=True)
            with Image.open(p) as im:
                im.load()
                if name.startswith('fig-'): shutil.copyfile(p,dest)
                else:
                    image=im.convert('RGB');image.thumbnail((1600,2400));image.save(dest,quality=90,optimize=True)
            assets[target]=p
        return url(target)

    cover=asset(book['cover'],ROOT/'README.md')
    epub_link=''
    if args.epub:
        p=ROOT/args.epub
        with zipfile.ZipFile(p) as z:
            if z.read('mimetype')!=b'application/epub+zip' or z.testzip(): raise ValueError('invalid EPUB')
            expected=len(docs)+1
            xhtml=[n for n in z.namelist() if n.endswith('.xhtml') and not n.endswith('nav.xhtml')]
            if len(xhtml)!=expected: raise ValueError('EPUB does not contain the expected chapter count')
        dest=out/'downloads/GPU-Modern-Computing.epub';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        epub_link=f'<a class="button" href="{url("downloads/GPU-Modern-Computing.epub")}">下载首版 EPUB · 2026-10-25</a>'

    def nav(active=''):
        fragments=[f'<a class="{"active" if active=="guide" else ""}" href="{url("guide/")}">导读与阅读路线</a>']
        for part in book['parts']:
            fragments.append('<div class="part-title">'+html.escape(part['title'])+'</div>')
            for ident in part['chapters']:
                d=next(x for x in docs if x['id']==ident)
                fragments.append(f'<a class="{"active" if active==ident else ""}" href="{url(d["slug"]+"/")}">{d["order"]:02}　{html.escape(d["title"])}</a>')
        fragments.append('<div class="part-title">附录与实验</div>')
        for d in docs[-len(book['appendices']):]:
            fragments.append(f'<a class="{"active" if active==d["id"] else ""}" href="{url(d["slug"]+"/")}">{html.escape(d["title"])}</a>')
        return ''.join(fragments)

    dialog='''<dialog id="search-dialog" aria-label="搜索全书"><div class="search-head"><input id="search-input" type="search" aria-label="搜索词" placeholder="搜索算子、显存、FSDP、低精度…" autocomplete="off"/><button id="search-close" aria-label="关闭搜索">×</button></div><p id="search-status" class="search-status">输入至少两个字或字母，搜索全书正文。</p><div id="search-results" class="search-results"></div></dialog>'''
    def page(title,content,description='从硬件、训练到算子优化，一部现代 AI 计算的中文图解知识书。'):
        return f'''<!doctype html><html lang="zh-CN" data-base="{base}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="{html.escape(description,quote=True)}"><meta name="theme-color" content="#fbf8f2"><title>{html.escape(title)} · GPU 与现代 AI 计算</title><link rel="stylesheet" href="{url('assets/style.css')}"><script defer src="{url('assets/app.js')}"></script></head><body><a class="skip" href="#main">跳到正文</a><header class="topbar"><nav class="top-inner" aria-label="主导航"><a class="brand" href="{url()}"><span>GPU</span>现代 AI 计算</a><a href="{url('#contents')}">全书目录</a><a class="repo-link" href="{REPO}">GitHub</a><button class="search-button" data-open-search>搜索全书</button></nav></header>{content}{dialog}<footer class="footer">GPU 与现代 AI 计算 · 2026年10月版　/　<a href="{REPO}">书稿与实验代码</a>　/　资料核验至 2026-10-25</footer></body></html>'''

    search=[]
    for at,d in enumerate(docs):
        source=ROOT/d['source'];text=source.read_text()
        if any(ord(c)<32 and c not in '\n\r\t' for c in text): raise ValueError('control character in '+source.name)
        if text.count('```')%2: raise ValueError('unbalanced fence in '+source.name)
        text=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',lambda m:'!['+m[1]+']('+asset(m[2],source)+')',text)
        maths=[]
        def math_token(m):
            raw=m[0];display=raw.startswith('$$');tex=raw[2:-2]if display else raw[1:-1]
            value=convert(tex.strip(),display='block'if display else'inline')
            ET.fromstring(value)
            token=f'GPUTEXPLACEHOLDER{len(maths):05}END';maths.append((token,value));return token
        parts=re.split(r'(```.*?```|`[^`\n]+`)',text,flags=re.S)
        for i in range(0,len(parts),2):parts[i]=re.sub(r'\$\$.*?\$\$|(?<!\$)\$[^\n$]+\$(?!\$)',math_token,parts[i],flags=re.S)
        rendered=markdown.markdown(''.join(parts),extensions=['tables','fenced_code','sane_lists','toc'],output_format='html')
        for token,value in maths:rendered=rendered.replace('<p>'+token+'</p>',value).replace(token,value)
        rendered=rendered.replace('<table>','<div class="table-wrap"><table>').replace('</table>','</table></div>')
        rendered=re.sub(r'<img ', '<img loading="lazy" decoding="async" ',rendered)
        # Bare source URLs are made clickable only in ordinary text, never inside code/anchors.
        pieces=re.split(r'(<pre\b.*?</pre>|<code\b.*?</code>|<a\b.*?</a>|<[^>]+>)',rendered,flags=re.S)
        for i in range(0,len(pieces),2):
            pieces[i]=re.sub(r'https?://[^\s<>]+',lambda m:'<a href="'+html.escape(html.unescape(m[0].rstrip('。，；）')),quote=True)+'">'+m[0].rstrip('。，；）')+'</a>'+m[0][len(m[0].rstrip('。，；）')):],pieces[i])
        rendered=''.join(pieces)
        previous=docs[at-1]if at else None;following=docs[at+1]if at+1<len(docs)else None
        prev=f'<a href="{url(previous["slug"]+"/")}"><span>上一章</span>{html.escape(previous["title"])}</a>'if previous else'<span></span>'
        nxt=f'<a href="{url(following["slug"]+"/")}"><span>下一章</span>{html.escape(following["title"])}</a>'if following else f'<a href="{url()}">返回全书目录</a>'
        content=f'''<div class="layout"><aside class="sidebar" aria-label="全书目录">{nav(d['id'])}</aside><main id="main" class="reader"><details class="mobile-toc"><summary>展开全书目录</summary><nav>{nav(d['id'])}</nav></details><div class="toolbar">字号 <button data-font="17">小</button><button data-font="18">中</button><button data-font="20">大</button><button data-font="22">更大</button></div><div class="kicker">从硬件、训练到算子优化</div><article data-page="{d['slug']}/">{rendered}</article><nav class="chapter-nav" aria-label="章节导航">{prev}{nxt}</nav></main></div>'''
        folder=out/d['slug'];folder.mkdir();(folder/'index.html').write_text(page(d['title'],content))
        plain=re.sub(r'!\[[^\]]*\]\([^)]+\)','',source.read_text())
        plain=re.sub(r'```.*?```',' ',plain,flags=re.S);plain=re.sub(r'[#*`|\[\]]',' ',plain);plain=re.sub(r'\s+',' ',plain)
        search.append({'title':d['title'],'url':d['slug']+'/','text':plain})

    cards=[]
    for number,part in enumerate(book['parts'],1):
        entries=[]
        for ident in part['chapters']:
            d=next(x for x in docs if x['id']==ident)
            entries.append(f'<li><a href="{url(d["slug"]+"/")}"><span class="num">{d["order"]:02}</span><span>{html.escape(d["title"])}</span></a></li>')
        cards.append(f'<section class="part"><div class="part-label">PART {number:02}</div><h3>{html.escape(part["title"])}</h3><ol>{"".join(entries)}</ol></section>')
    appendix_links=''.join(f'<a class="button" href="{url(d["slug"]+"/")}">{html.escape(d["title"])}</a>'for d in docs[-len(book['appendices']):])
    home=f'''<main id="main" class="wrap"><section class="hero"><div class="hero-copy"><div class="eyebrow">AN ILLUSTRATED GUIDE · 2026 EDITION</div><h1>GPU 与现代<br>AI 计算</h1><p class="subtitle">从硬件、训练到算子优化</p><p class="lead">从一次模型训练出发，追踪张量怎样经过框架、编译器、内核与网络，最终在芯片和集群上执行。理解原理，也亲手观察、计算和验证。</p><div class="actions"><a class="button primary" href="{url('guide/')}">开始阅读 →</a><a class="button" href="#contents">浏览全书目录</a><a id="resume-reading" class="button" href="{url('guide/')}" hidden>继续上次阅读</a></div><div class="stats"><div><b>36</b>完整章节</div><div><b>37</b>中文教学配图</div><div><b>20</b>实验脚本</div></div></div><img class="hero-cover" width="1024" height="1536" src="{cover}" alt="GPU与现代AI计算书籍封面"></section><div class="section-head"><h2>选择一条阅读路线</h2><small>从认识系统，到亲手优化</small></div><div class="route-grid"><a class="route" href="{url('training-step/')}"><h3>建立整体认识</h3><p>从一次训练看张量、梯度、芯片和显存，再理解性能与规模。</p><span class="path">第1—14章 → 性能 → 集群</span></a><a class="route" href="{url('reproducible-environment/')}"><h3>把训练跑起来</h3><p>复现环境、管理显存，串起精度、编译和完整Transformer实验。</p><span class="path">训练基础 → 编译 → 贯穿实验</span></a><a class="route" href="{url('first-kernel/')}"><h3>亲手编写算子</h3><p>从向量加法走到矩阵乘法、归约、反向与现代内核语言。</p><span class="path">执行与布局 → 第19—28章</span></a></div><div id="contents" class="section-head"><h2>全书目录</h2><small>八篇 · 统一主稿 · 按章阅读</small></div><div class="toc-grid">{''.join(cards)}</div><div class="section-head"><h2>实验与随手查阅</h2></div><div class="actions">{appendix_links}</div><div class="note">配图包含12张生成模型制作的教学示意（含封面）与25张精确图解，均附中文说明。CPU训练与恢复已验证；GPU内核、多卡与集群路径尚未实测。每章保留来源与适用条件。</div><div class="section-head"><h2>离线阅读与代码</h2></div><div class="downloads">{epub_link}<a class="button" href="{REPO}/archive/refs/heads/main.zip">下载书稿与实验源码</a><a class="button" href="{REPO}/actions/workflows/export.yml">整书导出记录</a></div></main>'''
    (out/'index.html').write_text(page('首页',home))
    (out/'search.json').write_text(json.dumps(search,ensure_ascii=False,separators=(',',':')))
    (out/'assets').mkdir(exist_ok=True)
    for name in ['style.css','app.js']:shutil.copyfile(ROOT/'site'/name,out/'assets'/name)
    (out/'.nojekyll').write_text('')
    # Check every local link, including the repository subpath, without network requests.
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in {'src','href'}and value:self.links.append(value)
    checked=0
    for p in out.rglob('*.html'):
        parser=Links();parser.feed(p.read_text())
        for link in parser.links:
            parts=urlsplit(link)
            if parts.scheme or parts.netloc or not parts.path:continue
            ref=unquote(parts.path)
            if ref.startswith('/'):
                if not ref.startswith(base):raise ValueError('outside base path '+link)
                target=out/ref[len(base):]
            else:target=p.parent/ref
            if target.is_dir():target=target/'index.html'
            if not target.is_file():raise ValueError('broken link '+p.name+' '+link)
            checked+=1
    summary={'pages':len(docs)+1,'chapters':len(book['chapters']),'images':len(assets),'internal_links_checked':checked,'base_path':base,'epub_supplied':bool(args.epub)}
    (out/'site-build.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':run()
