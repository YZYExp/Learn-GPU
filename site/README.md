# 阅读站

网站直接读取 `book.json`、`chapters/` 与 `appendices/`，不复制正文。数学使用本地构建的 MathML，不依赖在线数学渲染服务；图片保持原比例，正文生成图在网页中优化为JPEG，精确图保留PNG。

```bash
python -m pip install -r site/requirements.txt
python scripts/build_site.py --base-path /Learn-GPU/ --epub downloads/GPU-Modern-Computing.epub
python -m http.server 8000 --directory build/site
```

本地用根路径预览时，将 `--base-path` 改为 `/`。也可不传 `--epub`；此时不显示直接EPUB下载，不读取其他目录中的旧阅读包。所有本地资源链接在构建时检查，缺图或断链即失败。

## 发布

`.github/workflows/pages.yml` 负责轻量网页构建与GitHub Pages部署，PR仅构建检查。只安装3个网页依赖，不安装PyTorch、Matplotlib、Pandoc、TeX或字体，也不运行生图和整书导出。

首次需要将仓库 Settings → Pages → Build and deployment → Source 设为 **GitHub Actions**。普通工作流令牌通常不能代替仓库管理员完成首次管理设置。正确设置后，向main推送相关输入即触发发布。

`.github/workflows/export.yml` 独立导出EPUB及合并Markdown，自动或手动运行，上传30天保留的Actions artifact（下载通常需要GitHub登录）。网页部署不依赖此工作流。

网站中的直接EPUB下载明确标为2026-10-25首版，对应已验证的 `downloads/GPU-Modern-Computing.epub`。未来修订书稿时，应按发布版本更新这个文件与页面的版本日期；新运行的导出结果也可从独立工作流取得，不能把首版下载称为所有后续改动的同步版。

## 编辑资料

`editorial/`、`references/research-*.md`、Git历史和字体没有复制到站点。网站仅包含阅读内容、必要资源、搜索索引与明确提供的阅读包。
