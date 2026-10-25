#!/usr/bin/env python3
"""Render exact Chinese teaching figures for chapters 2/3/5/6/8/9/10/12.

Requires Pillow and a Chinese font. Set GPU_BOOK_CJK_FONT or pass --font.
Images are conceptual relationships; no device dimensions or timings are inferred.
"""
from pathlib import Path
import argparse
import math
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/images'
BUILD = ROOT / 'build'
BG='#fbf8f2'; NAVY='#12384a'; TEAL='#167a85'; ORANGE='#cb7228'
PALE='#e5eeeb'; SAND='#f1e4d3'; LINE='#c6d2d2'; WHITE='#ffffff'; GREY='#617782'
W,H=1600,1200
FONT=None

def f(size):
    return ImageFont.truetype(str(FONT),size)

def text(d,xy,s,size=38,fill=NAVY,anchor=None):
    d.text(xy,s,font=f(size),fill=fill,anchor=anchor)

def lines(d,xy,ss,size=36,fill=NAVY,gap=16):
    x,y=xy
    for s in ss:
        text(d,(x,y),s,size,fill)
        y+=size+gap

def card(d,box,fill=WHITE,outline=LINE,r=24):
    d.rounded_rectangle(box,radius=r,fill=fill,outline=outline,width=2)

def arrow(d,a,b,fill=TEAL,width=6):
    d.line([a,b],fill=fill,width=width)
    angle=math.atan2(b[1]-a[1],b[0]-a[0]); length=18
    p1=(b[0]-length*math.cos(angle-.45),b[1]-length*math.sin(angle-.45))
    p2=(b[0]-length*math.cos(angle+.45),b[1]-length*math.sin(angle+.45))
    d.polygon([b,p1,p2],fill=fill)

def base(n,title,subtitle):
    im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)
    text(d,(70,42),f'图 {n:02d}  /  GPU 与现代 AI 计算',32,TEAL)
    text(d,(70,100),title,62)
    text(d,(70,188),subtitle,34,GREY)
    d.line((70,250,1530,250),fill=LINE,width=2)
    return im,d

def foot(d,s):
    d.line((70,1105,1530,1105),fill=LINE,width=2)
    text(d,(70,1132),s,30,GREY)

def save(im,n):
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/f'fig-{n:02}.png'
    temporary=OUT/f'.fig-{n:02}.tmp.png'
    im.save(temporary,format='PNG')
    with Image.open(temporary) as check:
        check.verify()
    os.replace(temporary,target)

def matrix(d,x,y,vals,cw=125,ch=95):
    for i,row in enumerate(vals):
        for j,v in enumerate(row):
            bx=(x+j*cw,y+i*ch,x+(j+1)*cw-8,y+(i+1)*ch-8)
            card(d,bx,PALE if v%2==0 else WHITE,r=12)
            text(d,((bx[0]+bx[2])/2,(bx[1]+bx[3])/2),str(v),44,anchor='mm')

def fig2():
    im,d=base(2,'转置改变视图，数字未必搬动','同一段底层存储，可以用不同的形状与步幅解释')
    card(d,(70,285,765,795)); card(d,(805,285,1530,795))
    text(d,(105,312),'原视图  [2, 3]',44)
    matrix(d,190,400,[[0,1,2],[3,4,5]],140,110)
    lines(d,(110,660),['步幅 [3, 1]','位置 = 3 × 行号 + 列号'],36)
    text(d,(845,312),'转置视图  [3, 2]',44)
    matrix(d,1010,385,[[0,3],[1,4],[2,5]],140,85)
    lines(d,(845,660),['步幅 [1, 3]','位置 = 行号 + 3 × 列号'],36)
    text(d,(70,837),'共享底层存储：按地址依次排列',38,TEAL)
    for j in range(6):
        x=300+j*170
        card(d,(x,905,x+154,1000),PALE if j%2==0 else WHITE,r=12)
        text(d,(x+77,952),str(j),46,anchor='mm')
    text(d,(70,1019),'步幅单位为“元素”；转置视图与连续复制是两件事。',36)
    foot(d,'对应第2章 · 稠密张量示例，起始偏移为0，未执行连续化复制。');save(im,2)

def fig3():
    im,d=base(3,'链式法则把局部导数组合起来','教学函数：z = xw，L = z²；取 x = 2，w = 3')
    nodes=[(70,305,470,510),(600,305,1000,510),(1130,305,1530,510)]
    for b in nodes:card(d,b,WHITE)
    lines(d,(110,337),['输入','x = 2，w = 3'],42)
    lines(d,(650,337),['乘法','z = 2 × 3 = 6'],42)
    lines(d,(1170,337),['平方','L = 6² = 36'],42)
    arrow(d,(480,407),(588,407));arrow(d,(1010,407),(1118,407))
    text(d,(70,562),'反向：损失对参数的影响，沿依赖关系回传',40,TEAL)
    card(d,(70,635,1530,795),PALE)
    text(d,(110,660),'局部导数',36,TEAL)
    text(d,(110,718),'∂L / ∂z = 2z = 12',44)
    text(d,(850,718),'∂z / ∂w = x = 2',44)
    card(d,(70,835,1530,1048),SAND)
    text(d,(110,866),'组合：∂L / ∂w = (∂L / ∂z) × (∂z / ∂w)',42)
    text(d,(110,949),'参数梯度 = 12 × 2 = 24',52,ORANGE)
    foot(d,'对应第3章 · 求梯度时把x视为固定输入；这张图不表示优化器更新。');save(im,3)

def fig5():
    im,d=base(5,'GPU 演进：可表达的计算逐步扩展','按计算方式梳理历史；后来的能力与先前能力可以共存')
    rows=[('固定功能图形流水线','顶点、光栅化、像素等既定阶段','固定步骤'),('可编程着色器','让图形阶段中的计算可以编程','程序 + 数据'),('通用 GPU 计算','直接表达线程、内存与数值计算','大规模并行'),('矩阵与系统协同','矩阵单元、低精度、编译与多卡互连','数据流协同')]
    for i,(title,desc,badge) in enumerate(rows):
        y=285+i*192
        card(d,(70,y,1530,y+163),WHITE)
        d.ellipse((100,y+43,176,y+119),fill=TEAL if i%2==0 else ORANGE)
        text(d,(138,y+81),str(i+1),40,WHITE,anchor='mm')
        text(d,(215,y+21),title,44)
        text(d,(215,y+89),desc,34,GREY)
        card(d,(1215,y+43,1485,y+119),PALE,r=14)
        text(d,(1350,y+81),badge,34,TEAL,anchor='mm')
    foot(d,'对应第5章 · 概念演进图，不是完整年表；各厂商、产品路线存在差异。');save(im,5)

def fig6():
    im,d=base(6,'网格描述逻辑工作，硬件分批调度','1000个元素，每块256线程 → 4个块，共1024个逻辑线程')
    for i in range(4):
        x=70+(i%2)*750; y=295+(i//2)*263
        card(d,(x,y,x+710,y+230),WHITE)
        text(d,(x+28,y+18),f'线程块 {i}  /  全局编号 {i*256}—{i*256+255}',33)
        for j in range(8):
            bx=x+32+j*79
            card(d,(bx,y+90,bx+65,y+153),PALE if i<3 else SAND,r=9)
            text(d,(bx+32,y+121),str(j),31,anchor='mm')
        lab='256个线程全部有效' if i<3 else '仅768—999有效；1000—1023屏蔽'
        text(d,(x+28,y+175),lab,32,ORANGE if i==3 else TEAL)
    card(d,(70,855,1530,1045),PALE)
    lines(d,(105,878),['全局编号 = 块编号 × 256 + 块内编号','NVIDIA：每块256线程 = 8个warp，每个warp有32线程','线程块分配到SM；图中编号不代表执行先后。'],35,gap=14)
    foot(d,'对应第6章 · 小方格仅示意块内线程0—7；本图采用NVIDIA执行模型。');save(im,6)

def bits(d,y,name,counts):
    text(d,(70,y+22),name,42)
    x=370; width=1120; total=sum(counts); cw=width/total
    colors=[ORANGE,TEAL,NAVY]
    for group,(count,c) in enumerate(zip(counts,colors)):
        for j in range(count):
            a=round(x);b=round(x+cw)-2
            d.rectangle((a,y,b,y+83),fill=c)
            x+=cw
    text(d,(370,y+107),f'1位符号  +  {counts[1]}位指数  +  {counts[2]}位显式尾数',36)

def fig8():
    im,d=base(8,'同为浮点数，范围与精度取舍不同','位域只说明数值表示；实际矩阵吞吐还取决于硬件和运算路径')
    bits(d,310,'FP32 · 32位',[1,8,23]);bits(d,535,'FP16 · 16位',[1,5,10]);bits(d,760,'BF16 · 16位',[1,8,7])
    for x,c,s in [(70,ORANGE,'符号'),(390,TEAL,'指数：影响范围'),(900,NAVY,'尾数：影响数值间距')]:
        d.rectangle((x,1000,x+32,1032),fill=c);text(d,(x+47,992),s,34)
    foot(d,'对应第8章 · 每行等宽展示位域；FP8/FP4的格式与缩放另见第13章。');save(im,8)

def fig9():
    im,d=base(9,'沿数据路径看一台训练服务器','简化拓扑：主机内存 → CPU与I/O → GPU显存；跨机还要经过网络')
    card(d,(70,300,1530,950),WHITE)
    text(d,(105,318),'一台服务器',36,TEAL)
    card(d,(120,405,490,560),PALE);lines(d,(170,426),['主机内存','批次与CPU状态'],38)
    card(d,(120,680,490,840),PALE);lines(d,(170,704),['CPU与I/O','数据准备、调度'],38)
    arrow(d,(305,570),(305,668));text(d,(340,590),'内存访问',32,GREY)
    card(d,(790,405,1175,565),PALE);lines(d,(840,427),['GPU 0 + 显存','参数、激活、梯度'],36)
    card(d,(790,680,1175,840),PALE);lines(d,(840,704),['GPU 1 + 显存','本地计算与状态'],36)
    arrow(d,(505,760),(775,760));text(d,(530,695),'PCIe路径',34,TEAL)
    arrow(d,(980,576),(980,668),ORANGE);text(d,(1020,594),'可选高速互连',30,ORANGE)
    d.line([(630,760),(630,480)],fill=TEAL,width=5)
    arrow(d,(630,480),(775,480))
    card(d,(1240,610,1475,815),SAND);lines(d,(1274,635),['网卡','跨机通信'],34)
    arrow(d,(1188,760),(1227,760),ORANGE)
    lines(d,(70,991),['图中省略PCIe交换芯片与CPU插槽；实际拓扑必须查询。','GPU之间是否能直接访问，由设备、链路和系统配置共同决定。'],34,gap=14)
    foot(d,'对应第9章 · 连接关系示意，线条长短与粗细不表示真实带宽或延迟。');save(im,9)

def fig10():
    im,d=base(10,'可复现环境，要逐层确认兼容关系','预构建框架运行环境，与自行编译扩展的工具链要求并不相同')
    stages=[('训练代码与数据','提交、配置、分词器、种子'),('框架与算子扩展','PyTorch构建、扩展版本与目标架构'),('运行时与计算库','CUDA / ROCm运行库、数学与通信库'),('设备驱动与系统','操作系统、驱动版本、设备可见性'),('实际硬件','GPU型号、内存、互连与功耗设置')]
    for i,(title,desc) in enumerate(stages):
        y=290+i*147
        card(d,(70,y,1100,y+125),WHITE if i%2==0 else PALE)
        text(d,(105,y+12),title,40)
        text(d,(105,y+68),desc,31,GREY)
        if i<4:arrow(d,(1140,y+62),(1140,y+209),TEAL,width=4)
    card(d,(1190,328,1530,777),SAND)
    lines(d,(1220,362),['编译自定义扩展','还需确认：','','本地编译器','Toolkit / SDK','编译选项','目标GPU架构'],31,gap=18)
    text(d,(1190,851),'保存版本来源',33,ORANGE)
    text(d,(1190,903),'不只保存包名',33,ORANGE)
    foot(d,'对应第10章 · 驱动支持版本、本地编译器版本、框架构建版本是不同记录项。');save(im,10)

def fig12():
    im,d=base(12,'显存账本：先写假设，再算容量','示例：10亿可训练参数；BF16参数与梯度，FP32优化器状态及主参数')
    vals=[2,2,8,4];cols=[TEAL,'#5f9a99',NAVY,ORANGE];labs=['参数','梯度','两个Adam状态','主参数副本']
    x=70
    for v,c in zip(vals,cols):
        width=1460*v/16
        d.rectangle((x,310,x+width-3,450),fill=c)
        text(d,(x+width/2,380),f'{v} GB',42,WHITE,anchor='mm');x+=width
    for i,(lab,c) in enumerate(zip(labs,cols)):
        x=70+i*375
        d.rectangle((x,495,x+28,523),fill=c);text(d,(x+42,485),lab,32)
    card(d,(70,575,1530,760),PALE)
    text(d,(105,599),'常驻状态合计：16 GB ≈ 14.9 GiB',48)
    text(d,(105,678),'若没有FP32主参数副本，这一项合计变为12 GB。',35)
    text(d,(70,810),'还要计入：',40,TEAL)
    labels=['保存的激活','算子工作区','通信缓冲','框架与分配开销']
    for i,lab in enumerate(labels):
        x=70+i*375
        card(d,(x,885,x+335,988),SAND)
        text(d,(x+167,936),lab,33,anchor='mm')
    text(d,(70,1031),'OOM取决于同一时刻的峰值，模型文件大小不能代替训练预算。',34)
    foot(d,'对应第12章 · 教学假设，未采用分片/卸载；GB为十进制，GiB为二进制。');save(im,12)

def main():
    global FONT
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--font');args=ap.parse_args()
    candidates=[args.font,os.environ.get('GPU_BOOK_CJK_FONT'),str(ROOT/'assets/fonts/GPUAtlasSans.otf'),'/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']
    FONT=next((Path(p) for p in candidates if p and Path(p).is_file()),None)
    if FONT is None:raise SystemExit('Chinese font required: use --font PATH or GPU_BOOK_CJK_FONT')
    for fn in [fig2,fig3,fig5,fig6,fig8,fig9,fig10,fig12]:fn()
    BUILD.mkdir(exist_ok=True)
    contact=Image.new('RGB',(1600,600),BG)
    for j,n in enumerate([2,3,5,6,8,9,10,12]):
        im=Image.open(OUT/f'fig-{n:02}.png');im.thumbnail((400,300))
        contact.paste(im,((j%4)*400,(j//4)*300))
    contact_tmp=BUILD/'.hardware-contact.tmp.png'
    contact.save(contact_tmp,format='PNG')
    with Image.open(contact_tmp) as check:
        check.verify()
    os.replace(contact_tmp,BUILD/'hardware-contact.png')
    for n in [2,3,5,6,8,9,10,12]:
        with Image.open(OUT/f'fig-{n:02}.png') as check:
            check.verify()
    print(f'Rendered 8 figures; font={FONT}; contact=build/hardware-contact.png')

if __name__=='__main__':main()
