# 绘图字体

`GPUAtlasSans.otf` 是 Noto Sans CJK SC Regular 的字符子集，内部家族名改为 GPU Atlas Sans，约0.56MB，覆盖当前书稿与绘图脚本用字。原图不受字体子集影响；网页与EPUB也不加载这个绘图字体。

- 上游：https://github.com/notofonts/noto-cjk
- 原始字体：https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf
- 许可：同目录 `OFL.txt`，SIL Open Font License 1.1。
- 子集方法：fontTools.subset，保留当前Markdown/Python/JSON/CSS中的字符及可打印ASCII，保留名称与布局信息，再修改家族名。

重绘现有图可直接使用随附子集；如果新增字符缺字，可从官方源获取完整字体并通过 `--font` 参数指定，具体接口见各绘图脚本。
