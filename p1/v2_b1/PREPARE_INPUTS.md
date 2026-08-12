# P1 v2_b1 输入清单准备

`build_p1_v2_b1.py` 需要 9 个源清单：speech、noise、rir 各自的 train/dev/holdout。仓库中的 `prepare_split_manifests.py` 会直接索引已有 AISHELL、MUSAN 和 RIRS 音频，不复制或修改源音频。

```powershell
python p1/v2_b1/prepare_split_manifests.py `
  --aishell-root funasr_project/data/public/aishell1/extracted/data_aishell/wav_expanded `
  --musan-root funasr_project/data/public/augmentations/musan/extracted/musan `
  --rir-root funasr_project/data/public/augmentations/rirs_noises/extracted/RIRS_NOISES `
  --output funasr_project/data/p1_v2_b1_inputs
```

语音按 speaker 划分，确保 train/dev/holdout 人员不重叠且每个集合至少两名 speaker；MUSAN 与 RIRS 按现有 `PARAMETERS.json` 约定的相对路径散列划分。准备完成后可直接运行预检：

```powershell
python p1/v2_b1/build_p1_v2_b1_portable.py `
  --inputs funasr_project/data/p1_v2_b1_inputs `
  --source-root funasr_project `
  --output funasr_project/data/p1_v2_b1_preflight `
  --workers 2 `
  --preflight
```

预检会真正生成 14 条覆盖四种正式 split/场景的音频，不只是检查路径。Windows 使用 `spawn` worker，Linux 使用 `fork`。可移植入口仅适配本机数据目录和进程启动方式；冻结的 `build_p1_v2_b1.py` 保持不变。
