# P2 机器建档（machine inventory）

建档日期：2026-07-25　用途：P2 TSE 开发环境基线，云端环境锁（P2-04）的对照基准

## 硬件 / 系统

| 项 | 值 |
|---|---|
| OS | Windows（本机，仅用于开发/smoke；正式训练按手册须 Linux/云 GPU） |
| Python | 3.10.9 |
| torch | 2.13.0+cpu |
| CUDA | **不可用**（torch.cuda.is_available() = False） |
| D: 盘余量 | ~57 GB（2026-07-25 清理后） |
| 显存预算 | 任务书要求 6 GB 可运行（本机无法验证，见 blocked） |

## 能力边界（P2-04 FAIL 条款适用）

- 本机可做：API 形状开发、随机张量 smoke（CPU）、数据脚本编写
- 本机不可做：显存验收（<6GB）、100 步过拟合（P2-09）、正式训练（P2-10）
- GPU 机器到位后第一件事：`python tests/p2/smoke_tse_random.py --device auto` 补显存验收

## 已装关键依赖

| 包 | 版本 | 用途 |
|---|---|---|
| torch | 2.13.0+cpu | 模型/smoke |
| PyYAML | 已装 | 配置读取 |
| psutil | 未装 | CPU 内存检查项暂以 N/A 记录（不引手册外依赖） |

## 数据资产现状

| 资产 | 状态 | 位置 |
|---|---|---|
| MUSAN（music/noise/speech 11.73 GB） | 已解压 ✓ | data/public/augmentations/musan/extracted |
| RIRS_NOISES（3.52 GB） | 已解压 ✓ | data/public/augmentations/rirs_noises/extracted |
| AISHELL-1 train（14.64 GB，含 57 个 .tar 待展开） | 已下载 | data/public/aishell1/extracted |
| datasetA（0.31 GB） | 就绪 | data/datasetA |
| trials（4 说话人 enroll/clean/overlap） | 就绪 | data/trials |

## 网络

- 访问 GitHub / 外网需显式代理：`http://127.0.0.1:7897`（git 已全局配置）
