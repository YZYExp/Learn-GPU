#!/usr/bin/env python3
"""Generate chapter 14–18, 20–21, 23 explanatory PNGs with Pillow.

Run from any directory: python3 scripts/figures-compiler.py [--font /path/font.otf]
Requires Pillow and an installed CJK font. Does not generate measured data.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import argparse, math, os

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/images'
BG='#fbf8f2'; NAVY='#12384a'; TEAL='#167a85'; ORANGE='#cb7228'
WHITE='#ffffff'; PALE='#e6f0ed'; WARM='#f8e9d9'; LINE='#d9dfd9'; MUTED='#526976'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--font',default=os.environ.get('BOOK_CJK_FONT',''))
args=parser.parse_args()
candidates=[args.font,str(ROOT/'assets/fonts/GPUAtlasSans.otf'),'/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
'/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',str(ROOT/'build/fonts/GPUAtlasSans.otf')]
FONT=next((Path(x) for x in candidates if x and Path(x).is_file()),None)
if FONT is None:
    raise SystemExit('Chinese font required: pass --font PATH or set BOOK_CJK_FONT.')
font_cache={}
def font(size):
    if size not in font_cache: font_cache[size]=ImageFont.truetype(str(FONT),size)
    return font_cache[size]
def canvas(ch,title,subtitle,h=1200):
    im=Image.new('RGB',(1600,h),BG); d=ImageDraw.Draw(im)
    d.rounded_rectangle((64,45,185,96),radius=16,fill=NAVY)
    center(d,(124,70),f'第{ch}章',30,WHITE)
    d.text((64,116),title,font=font(55),fill=NAVY)
    d.text((67,192),subtitle,font=font(31),fill=MUTED)
    return im,d

def center(d,xy,t,size=34,color=NAVY):
    box=d.textbbox((0,0),t,font=font(size)); w=box[2]-box[0]; h=box[3]-box[1]
    d.text((xy[0]-w/2,xy[1]-h/2-box[1]),t,font=font(size),fill=color)
def txt(d,xy,t,size=34,color=NAVY,spacing=12):
    d.multiline_text(xy,t,font=font(size),fill=color,spacing=spacing)
def panel(d,box,fill=WHITE,outline=LINE,r=24):
    d.rounded_rectangle(box,radius=r,fill=fill,outline=outline,width=2)
def pill(d,box,t,fill=PALE,color=TEAL,size=32):
    panel(d,box,fill,fill,16);center(d,((box[0]+box[2])/2,(box[1]+box[3])/2),t,size,color)
def arrow(d,p,q,color=TEAL,width=5,head=14):
    d.line((p,q),fill=color,width=width)
    a=math.atan2(q[1]-p[1],q[0]-p[0]);d.polygon([q,(q[0]-head*math.cos(a-.45),q[1]-head*math.sin(a-.45)),(q[0]-head*math.cos(a+.45),q[1]-head*math.sin(a+.45))],fill=color)
def footer(d,h,t):
    d.line((64,h-105,1536,h-105),fill=LINE,width=2)
    txt(d,(66,h-83),t,30,MUTED)
def save(im,ch):
    OUT.mkdir(parents=True,exist_ok=True)
    target=OUT/f'fig-{ch:02}.png'
    staged=OUT/f'.fig-{ch:02}-compiler-tmp.png'
    im.save(staged,format='PNG',optimize=True)
    staged.replace(target)

def fig14():
    im,d=canvas(14,'用显存换计算，与拆分批次是两种选择','先看保留什么，再看何时更新参数。',1300)
    panel(d,(64,270,1536,690))
    txt(d,(100,300),'重计算：少保存内部激活，需要时重新计算',38,TEAL)
    xs=[120,590,1060]
    for i,x in enumerate(xs):
        pill(d,(x,385,x+390,480),f'模块{chr(65+i)} 前向',PALE,TEAL,35)
        pill(d,(x,515,x+390,583),'保存边界输入',BG,NAVY,32)
        if i<2:arrow(d,(x+395,432),(x+455,432))
    txt(d,(115,618),'反向到达某区域 → 重跑相关前向，取回求导所需的值',32)
    panel(d,(64,725,1536,1150))
    txt(d,(100,755),'梯度累积：多个微批次，合成一次参数更新',38,ORANGE)
    for i in range(4):
        x=110+i*365
        pill(d,(x,844,x+315,920),f'微批次{i+1}',WARM,ORANGE,34)
        center(d,(x+157,962),'计算梯度 ÷ 4',31)
        if i<3:center(d,(x+340,964),'+',40,ORANGE)
    arrow(d,(800,994),(800,1030),ORANGE)
    center(d,(800,1075),'梯度求和 → 更新一次参数',38,ORANGE)
    footer(d,1300,'累积示例：4个等大微批次、每批损失取平均；两种技术可以组合。')
    save(im,14)

def fig15():
    im,d=canvas(15,'三种“图”，各自回答不同问题','求导关系、程序优化与设备提交，不能混为一谈。',1300)
    for top,name,question,color in [(270,'求导图','损失怎样影响输入与参数？',TEAL),(550,'编译图','计算能怎样重写与融合？',ORANGE),(830,'设备执行图','设备任务怎样成组重放？',TEAL)]:
        panel(d,(64,top,1536,top+240));txt(d,(100,top+35),name,40,color);txt(d,(100,top+105),question,31)
    for j,t in enumerate(['输入 x','平方','损失求和']):
        x=740+j*250; pill(d,(x,310,x+210,385),t,PALE,TEAL,32)
        if j<2:arrow(d,(x+214,347),(x+242,347))
    arrow(d,(1430,430),(760,430),ORANGE);center(d,(1090,469),'沿依赖反向传播梯度',31,ORANGE)
    panel(d,(718,585,1495,735),WARM,WARM)
    for j,t in enumerate(['乘法','加法','激活']):
        x=750+j*250;pill(d,(x,610,x+200,675),t,WHITE,NAVY,33)
        if j<2:arrow(d,(x+205,643),(x+240,643),ORANGE)
    center(d,(1100,706),'可以形成一个融合内核',31,ORANGE)
    for j,t in enumerate(['内核甲','内核乙','复制任务']):
        x=740+j*250;pill(d,(x,870,x+210,940),t,PALE,TEAL,32)
        if j<2:arrow(d,(x+215,905),(x+242,905))
    center(d,(1090,1015),'保留任务依赖，减少重复提交开销',31)
    footer(d,1300,'示意三种职责；编译优化与设备执行图可以同时使用。')
    save(im,15)

def fig16():
    im,d=canvas(16,'图断裂与重编译：两个不同的边界','一个切开计算区域，一个改变已有方案的适用条件。',1300)
    panel(d,(64,270,1536,650))
    txt(d,(100,303),'图断裂：一次执行被分成多个区域',39,ORANGE)
    for x,lab,fill,col in [(130,'已捕获图甲',PALE,TEAL),(635,'普通Python逻辑',WARM,ORANGE),(1140,'已捕获图乙',PALE,TEAL)]:
        pill(d,(x,405,x+330,510),lab,fill,col,32)
    arrow(d,(473,458),(620,458),ORANGE);arrow(d,(979,458),(1125,458),ORANGE)
    txt(d,(120,560),'默认模式可在边界回到Python，再继续捕获；不等于结果错误。',32)
    panel(d,(64,690,1536,1145))
    txt(d,(100,723),'重编译：已有方案的条件不再满足',39,TEAL)
    pill(d,(120,825,530,910),'输入：长度128',PALE,TEAL,34)
    arrow(d,(548,866),(700,866));pill(d,(720,825,1460,910),'守卫通过 → 复用方案甲',PALE,TEAL,34)
    pill(d,(120,960,530,1045),'输入：长度256',WARM,ORANGE,34)
    arrow(d,(548,1001),(700,1001),ORANGE);pill(d,(720,960,1460,1045),'若原条件不适用 → 准备新方案',WARM,ORANGE,32)
    footer(d,1300,'形状变化不必然重编译；符号形状可让同一方案覆盖多个尺寸。')
    save(im,16)

def fig17():
    im,d=canvas(17,'融合减少的是中间结果的显存往返','同一公式：y = 最大值(a × x + b, 0)；a、b为标量。',1300)
    panel(d,(64,270,775,1115));panel(d,(825,270,1536,1115))
    center(d,(420,325),'三个独立内核',39,NAVY);center(d,(1180,325),'一个融合内核',39,TEAL)
    steps=[('乘法','读x → 写临时结果甲'),('加法','读甲 → 写临时结果乙'),('激活','读乙 → 写最终结果y')]
    for i,(lab,detail) in enumerate(steps):
        y=410+i*175
        pill(d,(110,y,730,y+80),lab,WARM,ORANGE,36)
        center(d,(420,y+123),detail,30)
        if i<2:arrow(d,(420,y+145),(420,y+169),ORANGE,4,10)
    center(d,(420,1005),'每步读写：8N字节',33,ORANGE)
    center(d,(420,1060),'三步合计：24N字节',37,ORANGE)
    pill(d,(900,420,1460,500),'从显存读取x',PALE,TEAL,34)
    arrow(d,(1180,520),(1180,580))
    panel(d,(900,595,1460,790),PALE,PALE)
    center(d,(1180,650),'乘法 → 加法 → 激活',33,TEAL)
    center(d,(1180,729),'中间值在内核内部使用',31,TEAL)
    arrow(d,(1180,810),(1180,870))
    pill(d,(900,890,1460,970),'向显存写出y',PALE,TEAL,34)
    center(d,(1180,1060),'合计：8N字节',39,TEAL)
    footer(d,1300,'教学假设：N个FP32元素，无缓存复用；忽略标量与其他流量，不等于3倍加速。')
    save(im,17)

def fig18():
    im,d=canvas(18,'从数学程序走向设备执行','以下以JAX路径为例；名字相近，承担的职责不同。',1300)
    stages=[('Python函数','用数组运算表达计算',PALE,TEAL),('JAX追踪与函数变换','观察计算，处理求导、批量化等',PALE,TEAL),('StableHLO','具有明确语义的高层操作表示',WARM,ORANGE),('XLA编译','选择布局、优化与生成目标程序',WARM,ORANGE),('设备执行程序','面向CPU、GPU或其他加速器',PALE,TEAL)]
    for i,(name,desc,fill,col) in enumerate(stages):
        y=270+i*155
        panel(d,(100,y,1170,y+120),fill,fill)
        txt(d,(135,y+18),name,36,col);txt(d,(135,y+69),desc,31)
        if i<4:arrow(d,(630,y+125),(630,y+148),col,4,11)
    panel(d,(1220,480,1536,900),WHITE,LINE)
    center(d,(1378,535),'MLIR',43,NAVY)
    for j,s in enumerate(['多层中间表示','与变换基础设施','可承载不同层次','及不同方言']):center(d,(1378,620+j*59),s,31,NAVY)
    footer(d,1300,'层次示意：MLIR不是另一种训练框架，也不是必须依次调用的额外阶段。')
    save(im,18)

def fig20():
    im,d=canvas(20,'能访问同一地址，不代表已经同步','访问范围、执行顺序与内存可见性，需要分别建立。',1400)
    panel(d,(64,270,1536,550));txt(d,(100,300),'块内屏障：参与同一线程块的合作',38,TEAL)
    for x,name,fill,col in [(120,'线程块甲',PALE,TEAL),(910,'线程块乙',WARM,ORANGE)]:
        panel(d,(x,375,x+570,485),fill,fill);center(d,(x+285,410),name,34,col)
        center(d,(x+285,454),'各自按规则执行块内屏障',30,col)
    center(d,(800,430),'≠',55,ORANGE)
    panel(d,(64,585,1536,850));txt(d,(100,615),'跨块访问：全局内存不会自动提供全局屏障',38,ORANGE)
    pill(d,(120,705,470,785),'块甲写入',WARM,ORANGE,34)
    pill(d,(625,705,975,785),'全局内存',PALE,TEAL,34)
    pill(d,(1130,705,1480,785),'块乙读取',WARM,ORANGE,34)
    arrow(d,(485,745),(610,745),ORANGE);arrow(d,(990,745),(1115,745),ORANGE)
    panel(d,(64,885,1536,1235));txt(d,(100,915),'跨流依赖：生产者记录事件，消费者等待事件',38,TEAL)
    txt(d,(115,1016),'流甲',34,TEAL);pill(d,(300,995,800,1075),'生产内核 → 记录事件',PALE,TEAL,32)
    txt(d,(115,1145),'流乙',34,ORANGE);pill(d,(900,1120,1465,1200),'等待事件 → 消费内核',WARM,ORANGE,32)
    arrow(d,(810,1035),(900,1160),TEAL)
    footer(d,1400,'普通内核跨块协作需明确机制；图中连线表示数据或事件关系，不代表同时执行。')
    save(im,20)

def fig21():
    im,d=canvas(21,'同一段存储，两种二维步长','地址偏移 = 行索引 × 行步长 + 列索引 × 列步长',1300)
    center(d,(800,300),'底层元素按地址顺序保存，转置视图不移动数据',34,NAVY)
    for i in range(6):
        x=185+i*205
        fill=PALE if i==3 else WARM if i==1 else WHITE
        panel(d,(x,365,x+180,445),fill,fill if fill!=WHITE else LINE)
        center(d,(x+90,404),str(i),40,TEAL if i==3 else ORANGE if i==1 else NAVY)
        center(d,(x+90,480),f'偏移{i}',30,MUTED)
    for bx,by,vals,name,shape,stride,col in [(120,635,[[0,1,2],[3,4,5]],'原始视图','形状 [2, 3]','步长 [3, 1]',TEAL),(940,600,[[0,3],[1,4],[2,5]],'转置视图','形状 [3, 2]','步长 [1, 3]',ORANGE)]:
        center(d,(bx+260,555),name,40,col)
        for r,row in enumerate(vals):
            for c,val in enumerate(row):
                x=bx+c*170;y=by+r*100
                chosen=r==1 and c==0
                panel(d,(x,y,x+150,y+80),PALE if chosen and col==TEAL else WARM if chosen else WHITE,LINE)
                center(d,(x+75,y+40),str(val),39,col if chosen else NAVY)
        center(d,(bx+260,950),shape,34,col)
        center(d,(bx+260,1008),stride,34,col)
    pill(d,(100,1080,760,1150),'位置(1,0)：偏移1×3+0×1=3',PALE,TEAL,30)
    pill(d,(840,1080,1500,1150),'位置(1,0)：偏移1×1+0×3=1',WARM,ORANGE,30)
    footer(d,1300,'步长以元素为单位；转换为字节地址时，还要乘每个元素的字节数。')
    save(im,21)

def fig23():
    im,d=canvas(23,'稳定Softmax：先减最大值，再取指数','有限输入示例；所有概率共同归一化，保持数学上的比例。',1300)
    xs=[1000,1001,1002];shift=[x-max(xs) for x in xs]
    exps=[math.exp(x) for x in shift];s=sum(exps);probs=[x/s for x in exps]
    cols=[715,1030,1345]
    for i,x in enumerate(cols):center(d,(x,310),f'第{i+1}项',34,MUTED)
    rows=[('原始输入',list(map(str,xs)),WHITE,NAVY),('减去最大值1002',list(map(str,shift)),WARM,ORANGE),('对结果取指数',[f'{x:.4f}' for x in exps],PALE,TEAL),('除以指数之和',[f'{x:.4f}' for x in probs],PALE,TEAL)]
    for i,(lab,values,fill,col) in enumerate(rows):
        y=370+i*155
        panel(d,(64,y,1536,y+115),fill,LINE)
        txt(d,(100,y+35),lab,34,col)
        for x,value in zip(cols,values):center(d,(x,y+58),value,40,col)
    pill(d,(100,1040,1500,1120),f'分母约为{s:.4f}；概率之和约为1',WARM,ORANGE,36)
    footer(d,1300,'数值为近似显示；全行被掩为负无穷时，需要额外定义与处理。')
    save(im,23)

if __name__=='__main__':
    for fn in [fig14,fig15,fig16,fig17,fig18,fig20,fig21,fig23]:fn()
    thumbs=[]
    for n in [14,15,16,17,18,20,21,23]:
        im=Image.open(OUT/f'fig-{n:02}.png').convert('RGB');im.thumbnail((640,560))
        slot=Image.new('RGB',(660,590),BG);slot.paste(im,((660-im.width)//2,15));thumbs.append(slot)
    sheet=Image.new('RGB',(1320,2360),LINE)
    for k,im in enumerate(thumbs):sheet.paste(im,((k%2)*660,(k//2)*590))
    (ROOT/'build').mkdir(exist_ok=True);sheet.save(ROOT/'build/compiler-contact.jpg',quality=90)
    print('Generated 8 chapter figures and build/compiler-contact.jpg')
