# 上传说明

把**本目录**建成独立仓库，不要推送整个实验工程。

```bash
cd fetal-chd-gated-triage
git init
git add .
git status   # 确认没有 .pt / 影像 / jsonl 缓存 / .cursor
git commit -m "Initial public code drop for gated four-view fetal CHD triage."
```

然后在 GitHub 新建空仓库，按页面提示 `git remote add` + `push`。

`.gitignore` 已排除 `*.pt`、`*.jpg`、`data/`、`outputs/`，以及 `.cursor/` 等编辑器痕迹。

**没有放：** 切面检测器权重、私有超声、FetalCLIP 权重、CARDIUM 像素。  
**论文融合表**走 `--embed-load-only`（嵌入+标签缓存），不需要检测器 `.pt`。
