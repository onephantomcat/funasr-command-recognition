# WeSep 上游来源登记（P2-03 冻结证据）

- 冻结日期：2026-07-30
- 仓库：https://github.com/wenet-e2e/wesep
- 冻结 commit（40 位）：`99eca54b60300d39b9353d93cf285a14bba37854`
- commit 日期：2025-10-04 10:54:28 +0800（Merge pull request #33 from msinanyildirim/patch-1）
- 本地路径：`third_party/wesep/`（clone 后未做任何修改，`git status --porcelain` 为空，见 wesep_tree_status.txt）

## 文件哈希（SHA256）

```
9828687FFACCAD90769D96050154AE6BCDF6CBE737AA512888A047A097560FA0  README.md
20E19F81CE484B3F6A3F3579ADA692788FC57BF3B26FC14F356B342C250E7F1E  requirements.txt
0EC421E541F1593C379F21B132320AFB13C90310421E9D1C70C2835D69FA6CDF  setup.py
```

## LICENSE 状态

- 本 commit 仓库根目录**无 LICENSE 文件**（`Get-ChildItem -Filter LICENSE*` 无结果）
- GitHub 页面标注 Apache-2.0；`licenses/wesep/` 目录已建，待补官方 LICENSE 文本或截图存档

## requirements.txt 审计结论（面向本机 Windows + torch 2.7.1+cu126 + Py3.10）

| 依赖 | 风险 | 处置 |
|---|---|---|
| pesq==0.0.4 | Windows 无预编译 wheel，需 C 编译器 | **跳过**（仅评估指标用，冒烟不需要） |
| numpy==1.22.4 / scipy==1.7.3 | 旧钉版，与现环境 numpy 2.x 冲突 | 不安装钉版，用现环境版本 |
| librosa==0.10.1 | 依赖 numba，Py3.10 可用 | 按需（import 报错再装） |
| silero-vad / kaldiio / lmdb / mir_eval / pystoi / fast_bss_eval | 纯 Python 或轻量 | 按需 |
| torchaudio | **requirements 未列出但代码必需** | 需装 torchaudio 2.7.1+cu126（与 torch 匹配） |
| flake8* / pre-commit / matplotlib / thop / torchnet / tableprint / fire / joblib / tqdm / h5py / auraloss / torchmetrics | dev/训练辅助 | 冒烟阶段不装 |

## 冻结纪律

- 同一实验系列（B1）内禁止 `git pull` / 切换 commit
- 对上游的任何本地修改须另存 patch 并记录（当前无）
