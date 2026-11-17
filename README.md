# 最快单执行器基线

源码来自 gossip_ae_latest_20261112_160555/source/current，保存于 2026-11-17。
24 个原始源码文件按 source_hashes.json 逐个核验通过。requirements.txt 从当前项目复制，作为运行依赖说明。

本版本包含普通转发按 key 覆盖、AE 补发按槽位覆盖、重复请求抑制、请求写入前剪枝和待发摘要合并。使用单执行器 FIFO。

种子 20261104 的历史 T100 为 N=100 的 11.558372954513 s、N=300 的 11.984625029587 s、N=1000 的 16.716287156735 s。详见 historical_summary.json。

在本目录运行以下命令可将新结果写到本目录的 results 文件夹。

```powershell
python -B run_random_matrix.py --nodes 100 300 --seeds 20261104 --arms gossip --out results
```

本历史入口没有 executor-mode 或 executor-scheduler 参数。
