#!/usr/bin/env python3
"""Rebuild exact system teaching figures. Pillow + a CJK TrueType/OpenType font.

Run from any directory:
  python3 scripts/figures-systems.py --font /path/to/NotoSansCJK-Regular.ttc
No device performance data is invented. All figures are conceptual or explicitly
labelled mathematical models. Outputs are 1600 px wide with type >=30 px.
"""
from pathlib import Path
import argparse
import math
import os
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/images'
BG='#fbf8f2'; NAVY='#12384a'; TEAL='#167a85'; ORANGE='#cb7228'
MUTED='#526b73'; LINE='#cbd5d3'; PALE='#e5efeb'; WARM='#f3e3d4'; WHITE='#ffffff'
FONT=None


def font(size=36):
    return ImageFont.truetype(str(FONT), size)


def wrap(text, width, size):
    # Keep explicit line breaks; wrap by actual measured pixel width, incl. CJK.
    lines=[]
    for paragraph in text.split('\n'):
        line=''
        for ch in paragraph:
            if line and font(size).getlength(line+ch)>width:
                lines.append(line);line=ch
            else: line+=ch
        lines.append(line)
    return lines


def txt(d, xy, text, size=36, fill=NAVY, width=None, linegap=10, anchor=None):
    x,y=xy
    if width is None:
        d.text((x,y),text,font=font(size),fill=fill,anchor=anchor)
        return size+linegap
    lines=wrap(text,width,size)
    for line in lines:
        d.text((x,y),line,font=font(size),fill=fill)
        y+=size+linegap
    return len(lines)*(size+linegap)


def rr(d, box, fill=WHITE, outline=LINE, radius=22, width=2):
    d.rounded_rectangle(box,radius=radius,fill=fill,outline=outline,width=width)


def arrow(d,a,b,fill=TEAL,width=5,head=15):
    d.line([a,b],fill=fill,width=width)
    angle=math.atan2(b[1]-a[1],b[0]-a[0])
    points=[b,(b[0]-head*math.cos(angle-0.55),b[1]-head*math.sin(angle-0.55)),
              (b[0]-head*math.cos(angle+0.55),b[1]-head*math.sin(angle+0.55))]
    d.polygon(points,fill=fill)


def base(num,title,subtitle,height=1200):
    im=Image.new('RGB',(1600,height),BG);d=ImageDraw.Draw(im)
    txt(d,(80,40),f'GPU 与现代 AI 计算  /  第{num:02}章',30,TEAL)
    txt(d,(80,99),title,62)
    txt(d,(80,185),subtitle,34,MUTED,width=1440)
    d.line((80,255,1520,255),fill=LINE,width=3)
    return im,d


def footer(d,text,height=1200):
    d.line((80,height-104,1520,height-104),fill=LINE,width=2)
    txt(d,(80,height-82),text,30,MUTED,width=1440,linegap=5)


def save(im,num):
    path=OUT/f'fig-{num:02}.png'
    temp=path.with_suffix('.png.tmp')
    im.save(temp,format='PNG',optimize=True)
    with Image.open(temp) as check: check.verify()
    with temp.open('rb') as handle: os.fsync(handle.fileno())
    os.replace(temp,path)
    return path


def boxtext(d,box,title,body,accent=TEAL,body_size=35):
    rr(d,box)
    x,y,x1,y1=box
    d.rounded_rectangle((x,y,x+9,y1),radius=4,fill=accent)
    txt(d,(x+28,y+20),title,42,accent,width=x1-x-52)
    txt(d,(x+28,y+83),body,body_size,width=x1-x-52,linegap=12)


def fig24():
    im,d=base(24,'自定义算子：数值与契约一起正确','以逐元素 SiLU 为例：前向、反向、元信息与框架行为分别检查')
    boxtext(d,(80,295,770,500),'前向：计算输出','y = sigmoid(x) × x\n可保存 x，供反向重新计算',TEAL,37)
    boxtext(d,(830,295,1520,500),'反向：传回梯度','令 s = sigmoid(x)\ndx = dy × [s + x × s × (1 − s)]',ORANGE,35)
    arrow(d,(770,395),(830,395),ORANGE)
    txt(d,(80,545),'同一个算子，四组接口约束',43)
    items=[('元信息','形状、类型、设备\n真实与 Fake 一致'),('存储行为','是否修改输入\n是否与输入共享存储'),('数学正确','输出对照参考值\n梯度做数值检查'),('系统配合','编译、自动求导\n支持范围与回退')]
    for i,(a,b) in enumerate(items):
        x=80+i*365
        rr(d,(x,625,x+345,880),PALE if i%2==0 else WARM)
        txt(d,(x+24,650),a,41,TEAL if i%2==0 else ORANGE)
        txt(d,(x+24,725),b,34,width=297,linegap=15)
    rr(d,(80,930,1520,1045),NAVY,NAVY)
    txt(d,(110,958),'opcheck 检查注册契约；gradcheck 检查梯度数学。二者不能互相替代。',34,WHITE,width=1380)
    footer(d,'公式为实数逐元素 SiLU；具体设备、类型与布局须以实现声明为准。')
    return save(im,24)


def fig25():
    im,d=base(25,'Roofline：先辨认性能上限','教学假设：匹配运算类型的计算上限 20 TFLOP/s，带宽 1 TB/s')
    L,T,R,B=260,340,1480,875
    def X(v): return L+(math.log10(v)+2)/5*(R-L)
    def Y(v): return B-(math.log10(v)+2)/4*(B-T)
    for v in [.01,.1,1,10,100,1000]:
        x=X(v);d.line((x,T,x,B),fill=LINE,width=2)
        txt(d,(x,B+18),str(v) if v<1 else str(int(v)),30,MUTED,anchor='mt')
    for v in [.01,.1,1,10,100]:
        y=Y(v);d.line((L,y,R,y),fill=LINE,width=2)
        txt(d,(L-24,y),str(v) if v<1 else str(int(v)),30,MUTED,anchor='rm')
    d.line((L,T,L,B,R,B),fill=NAVY,width=4)
    txt(d,(80,287),'性能（TFLOP/s）',34)
    txt(d,(640,942),'算术强度（FLOP/Byte，对数轴）',34)
    d.line([(X(.01),Y(.01)),(X(20),Y(20)),(X(1000),Y(20))],fill=TEAL,width=10)
    txt(d,(920,355),'计算上限：20 TFLOP/s',35,TEAL)
    rr(d,(330,355,800,510),WHITE)
    txt(d,(355,375),'带宽线：带宽 × 算术强度\n屋顶线取两种上限的较小值',33,width=420)
    x,y=X(1/12),Y(1/12)
    d.ellipse((x-11,y-11,x+11,y+11),fill=ORANGE)
    arrow(d,(620,675),(x+12,y-5),ORANGE,4)
    txt(d,(640,615),'FP32 向量加法\n强度 ≈ 1/12 FLOP/Byte\n带宽屋顶 ≈ 83.3 GFLOP/s',34,ORANGE,width=560)
    txt(d,(1020,540),'受计算上限约束',34,TEAL)
    rr(d,(80,1000,1520,1065),PALE)
    txt(d,(105,1010),'同一精度、同一运算与同一存储层级，才能比较；屋顶线不是性能保证。',32,width=1390)
    footer(d,'图中是数学模型与假设参数，未使用任何产品跑分；横纵轴均为对数尺度。')
    return save(im,25)


def fig26():
    im,d=base(26,'异步流水线：搬运下一块，计算当前块','四个数据块；每块搬运 3 单位、计算 5 单位；忽略资源争用与同步成本')
    L=250;scale=37
    def bar(y,s,e,label,c):
        x=L+s*scale;x1=L+e*scale
        rr(d,(x,y,x1-3,y+82),c,c,12)
        txt(d,((x+x1)/2,y+40),label,32,WHITE,anchor='mm')
    txt(d,(80,308),'串行执行',42)
    for i in range(4):
        bar(395,i*8,i*8+3,f'搬{i+1}',TEAL)
        bar(395,i*8+3,i*8+8,f'算{i+1}',ORANGE)
    for v in [0,8,16,24,32]:
        txt(d,(L+v*scale,495),str(v),32,MUTED,anchor='mt')
    txt(d,(80,560),'双缓冲重叠',42)
    txt(d,(80,675),'搬运',36,TEAL)
    txt(d,(80,795),'计算',36,ORANGE)
    for i,s in enumerate([0,3,8,13]): bar(650,s,s+3,f'搬{i+1}',TEAL)
    for i in range(4): bar(770,3+i*5,8+i*5,f'算{i+1}',ORANGE)
    for v in [0,8,16,24,32]:
        txt(d,(L+v*scale,880),str(v),32,MUTED,anchor='mt')
    d.line((L+23*scale,635,L+23*scale,870),fill=NAVY,width=3)
    txt(d,(L+23*scale+20,690),'23 单位\n完成',36,width=280)
    rr(d,(80,965,1520,1055),PALE)
    txt(d,(105,984),'串行 32 单位 → 理想重叠 23 单位；先填充、再重叠、最后排空。',35,width=1390)
    footer(d,'槽位复用前必须等读取完成；实际阶段深度还受共享内存、寄存器和同步限制。')
    return save(im,26)


def fig28():
    im,d=base(28,'现代内核语言：控制权放在哪里','表达层级是一组设计取舍；本图不表示性能排名，也不保证跨平台等速',1320)
    rows=[('CUDA / HIP','线程、地址与同步','显式组织低层执行'),('Triton','数据块与块内运算','编译器完成底层映射'),('CuTe DSL','布局与硬件流水线','细致控制矩阵计算'),('cuTile','数据块与数组操作','底层并行由工具链映射'),('Helion','接近 PyTorch 的计算','结合自动配置搜索'),('Pallas / TileLang','各自的分块和存储模型','按后端核对支持范围')]
    xs=[80,420,980,1520];y=300;rh=116
    rr(d,(80,y,1520,y+77),NAVY,NAVY,12)
    for x,t in zip([105,445,1005],['工具路线','开发者主要表达','需要关注的边界']):txt(d,(x,y+17),t,35,WHITE)
    y+=89
    for i,(a,b,c) in enumerate(rows):
        rr(d,(80,y,1520,y+rh-8),WHITE if i%2==0 else PALE,LINE,12)
        txt(d,(105,y+29),a,35,TEAL)
        txt(d,(445,y+29),b,35)
        txt(d,(1005,y+29),c,34)
        y+=rh
    txt(d,(80,1120),'同一算子对照：正确性 → 编译与调优成本 → 内核延迟 → 端到端收益',35,ORANGE,width=1440)
    footer(d,'核验日期：2026-10-25。功能、后端及硬件要求随版本变化；CuTe DSL 与 cuTile 是不同项目。',1320)
    return save(im,28)


def fig30():
    im,d=base(30,'复制与分片：同一张量怎样保存','四张卡、四片参数作概念对照；这里只展示参数，不代表完整显存账本')
    txt(d,(80,300),'DDP：各卡保存完整副本',42,TEAL)
    txt(d,(850,300),'全分片：各卡常驻一片',42,ORANGE)
    colors=[TEAL,ORANGE,NAVY,'#648f82']
    for i in range(4):
        y=390+i*100
        txt(d,(80,y+16),f'卡{i}',35)
        txt(d,(850,y+16),f'卡{i}',35)
        for j,c in enumerate(colors):
            x=205+j*120;rr(d,(x,y,x+105,y+75),c,c,12)
            txt(d,(x+52,y+37),chr(65+j),36,WHITE,anchor='mm')
        for j,c in enumerate(colors):
            x=975+j*120
            rr(d,(x,y,x+105,y+75),c if j==i else BG,c if j==i else LINE,12)
            if j==i:txt(d,(x+52,y+37),chr(65+j),36,WHITE,anchor='mm')
    d.line((780,295,780,790),fill=LINE,width=3)
    rr(d,(80,845,1520,1030),PALE)
    txt(d,(105,867),'计算某个模块前',38,TEAL)
    txt(d,(105,930),'通过 AllGather 获得所需完整参数；按策略用后重分片。',36,width=1360)
    txt(d,(80,1052),'分片省常驻状态；临时组装、激活和通信缓冲仍会抬高峰值。',33,ORANGE)
    footer(d,'ZeRO 的各阶段分片对象不同；FSDP 的模块粒度、预取与精度会改变实际容量和通信。')
    return save(im,30)


def fig31():
    im,d=base(31,'并行切分：切的是不同维度','数据、层内张量、模型深度和上下文，形成不同的协作边界',1320)
    specs=[(80,300,'数据并行','同一模型，处理不同样本',TEAL),(830,300,'张量并行','同一层矩阵，分片协作',ORANGE),(80,755,'流水线并行','不同卡持有不同连续层',ORANGE),(830,755,'上下文并行','分段查询，仍需交换键值',TEAL)]
    for x,y,t,sub,c in specs:
        rr(d,(x,y,x+690,y+405),WHITE)
        txt(d,(x+25,y+22),t,42,c)
        txt(d,(x+25,y+85),sub,34,width=640)
    # DP: two identical model boxes receive distinct sample cards.
    for i in range(2):
        x=120+i*310
        rr(d,(x,455,x+250,515),PALE);txt(d,(x+125,485),f'样本组 {i+1}',33,anchor='mm')
        arrow(d,(x+125,523),(x+125,552),TEAL)
        rr(d,(x,560,x+250,635),TEAL,TEAL);txt(d,(x+125,597),'完整模型',35,WHITE,anchor='mm')
    # TP: W cut output dimensions; concatenate output shards.
    for i,c in enumerate([ORANGE,TEAL]):
        x=870+i*310
        rr(d,(x,455,x+250,545),c,c);txt(d,(x+125,500),f'权重列片 {i+1}',34,WHITE,anchor='mm')
        arrow(d,(x+125,552),(x+125,580),c)
        txt(d,(x+125,615),f'输出片 {i+1}',34,anchor='mm')
    # PP two stages.
    rr(d,(120,925,360,1020),ORANGE,ORANGE);txt(d,(240,972),'前半模型层',34,WHITE,anchor='mm')
    rr(d,(470,925,710,1020),TEAL,TEAL);txt(d,(590,972),'后半模型层',34,WHITE,anchor='mm')
    arrow(d,(375,960),(455,960),NAVY);arrow(d,(455,995),(375,995),ORANGE)
    txt(d,(120,1055),'前向传激活；反向传激活梯度',33,width=590)
    # CP two local Qs with KV exchange.
    for i,c in enumerate([TEAL,ORANGE]):
        x=870+i*310
        rr(d,(x,925,x+250,1015),c,c);txt(d,(x+125,970),f'本段查询 {i+1}',34,WHITE,anchor='mm')
    arrow(d,(1140,955),(1170,955),NAVY);arrow(d,(1170,990),(1140,990),NAVY)
    txt(d,(870,1055),'完整注意力需要其他片段的 KV',33,width=600)
    footer(d,'示意不保证均匀加速；完整模型副本用于数据并行，其他切分方式需要额外协作。',1320)
    return save(im,31)


def fig33():
    im,d=base(33,'能恢复训练，需要保存哪些状态','检查点记录一个逻辑时刻；模型权重文件只是其中一部分',1320)
    # State inventory around logical snapshot.
    rr(d,(80,300,1520,420),NAVY,NAVY)
    txt(d,(800,360),'同一步骤的完整快照',46,WHITE,anchor='mm')
    cards=[('模型参数','权重及分片布局'),('优化器状态','动量、一阶与二阶统计'),('训练进度','步数、学习率与缩放'),('随机状态','采样、Dropout 等'),('数据位置','已消费样本或 token'),('运行身份','代码、配置与数据版本')]
    for i,(a,b) in enumerate(cards):
        x=80+(i%3)*490;y=465+(i//3)*200
        rr(d,(x,y,x+460,y+170),PALE if i%2==0 else WARM)
        txt(d,(x+25,y+20),a,39,TEAL if i%2==0 else ORANGE)
        txt(d,(x+25,y+87),b,33,width=410)
    rr(d,(80,920,1520,1090),WHITE)
    txt(d,(110,940),'恢复验收',40,TEAL)
    txt(d,(110,1007),'读取 → 重建状态 → 继续训练 → 比较损失、参数与数据进度',35,width=1360)
    txt(d,(80,1130),'已暂存 ≠ 已落盘；文件存在 ≠ 已证明可恢复。',38,ORANGE)
    footer(d,'换卡数、换布局或换版本时，需要另行验证重分片、格式兼容和数值复现范围。',1320)
    return save(im,33)


def fig34():
    im,d=base(34,'训练方法不同，资源流也不同','比较计算结构；本图不以参数量推断显存，也不提供性能排名',1320)
    rows=[('预训练 / SFT','给定输入与目标','前向 → 反向 → 更新','有效 token、激活与吞吐'),('LoRA / QLoRA','基础权重 + 适配器','少量参数参与更新','基础权重与激活仍占内存'),('蒸馏','教师输出或分布','教师指导 → 学生更新','在线教师增加一条计算路径'),('DPO','已给定的偏好对','比较概率 → 更新策略','参考概率可预先计算'),('在线 RL 后训练','当前策略生成轨迹','生成 → 评价 → 更新','解码、环境长尾与权重同步')]
    widths=[300,380,380,380];xs=[80,380,760,1140]
    rr(d,(80,300,1520,385),NAVY,NAVY)
    for x,t in zip([100,400,780,1160],['工作负载','数据从哪里来','典型计算结构','额外系统成本']):txt(d,(x,322),t,33,WHITE)
    y=400
    for i,row in enumerate(rows):
        rr(d,(80,y,1520,y+135),WHITE if i%2==0 else PALE,LINE,12)
        for x,w,t in zip(xs,widths,row):txt(d,(x+20,y+30),t,33,TEAL if x==80 else NAVY,width=w-42,linegap=10)
        y+=150
    txt(d,(80,1172),'吞吐要连同生成成本、有效训练量与最终模型质量一起比较。',35,ORANGE,width=1440)
    footer(d,'各算法与实现存在变体；QLoRA 并非把所有前向、反向和优化器都改成四位运算。',1320)
    return save(im,34)


def fig36():
    im,d=base(36,'比较计算生态，要逐层核对','按任务验证“可装下、可运行、可优化、可扩展”，不用峰值代替系统表现',1400)
    rows=[('应用与模型','训练、推理、算子、精度和输入形状'),('框架与编译','PyTorch / JAX / MLX；图编译与后端'),('内核与库','数学库、定制内核、量化与算子覆盖'),('通信与运行','集合通信、检查点、调试与观测'),('硬件与系统','容量、带宽、互连、驱动与操作系统')]
    y=300
    for i,(a,b) in enumerate(rows):
        rr(d,(80,y,1520,y+100),WHITE if i%2==0 else PALE)
        txt(d,(110,y+26),a,39,TEAL)
        txt(d,(470,y+29),b,35,width=1010)
        y+=115
    txt(d,(80,910),'生态入口举例（不表示等价或完整支持）',40)
    platforms=[('NVIDIA','CUDA / cuTile'),('AMD','ROCm / HIP'),('Apple','Metal / MLX'),('Intel','oneAPI / XPU'),('Google TPU','JAX / XLA')]
    for i,(a,b) in enumerate(platforms):
        x=80+i*292
        rr(d,(x,990,x+272,1115),TEAL if i%2==0 else NAVY,TEAL if i%2==0 else NAVY,15)
        txt(d,(x+136,1020),a,34,WHITE,anchor='mm')
        txt(d,(x+136,1072),b,30,WHITE,anchor='mm')
    txt(d,(80,1180),'相同品牌、同一架构家族，也要检查具体型号与软件兼容矩阵。',35,ORANGE,width=1440)
    footer(d,'平台入口核验至 2026-10-25；选择应依据同样训练语义和质量目标下的实际任务验证。',1400)
    return save(im,36)


def main():
    global FONT
    parser=argparse.ArgumentParser();parser.add_argument('--font',type=Path)
    args=parser.parse_args()
    candidates=[args.font,ROOT/'assets/fonts/GPUAtlasSans.otf',Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),Path('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')]
    FONT=next((p for p in candidates if p and p.exists()),None)
    if FONT is None: raise SystemExit('Supply a Chinese font with --font /path/to/font.ttc')
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[f() for f in [fig24,fig25,fig26,fig28,fig30,fig31,fig33,fig34,fig36]]
    # Internal QA contact sheet only; original figures retain large type.
    contact=Image.new('RGB',(1600,1460),BG);dd=ImageDraw.Draw(contact)
    for i,p in enumerate(paths):
        thumb=Image.open(p);thumb.thumbnail((510,430))
        x=20+(i%3)*530;y=20+(i//3)*480
        contact.paste(thumb,(x,y))
        txt(dd,(x,y+435),p.name,30)
    (ROOT/'build').mkdir(exist_ok=True)
    contact.save(ROOT/'build/systems-contact.jpg',quality=92)
    for p in paths:print(p.relative_to(ROOT))

if __name__=='__main__':main()
