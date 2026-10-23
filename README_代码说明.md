# ESPC-Transolver 相关实验：仅源码包

按用户最终要求，本目录只保留源码、配置、依赖说明与实验协议。
不含训练/测试数据缓存、原始样本、已训练权重、训练结果或 Python/Torch 安装环境。
原工程及全部数据保持原样；此前临时复制的数据和环境副本已从本目录移除。

## 五个方法的入口

| 方法 | 实验标识 | 模型 / 输入实现 | 训练入口 |
|---|---|---|---|
| ESPC-Transolver | M0 | src/cspgnet/v3_compositional_transolver.py；v3_object_transolver.py | scripts/train_v6s11_20k_all_methods.py --method M0 |
| Transolver-Geo-strong | M5 | scripts/train_v6s11_20k_all_methods.py 中 M5CompositionalGeoStrong | 同一脚本 --method M5 |
| Transolver-Geo-matched | M4 | 同一组合式主干；PreparedScenes 用占据编码替换输入 | 同一脚本 --method M4 |
| PIBNet | M3 | src/pibnet/gnn/pibnet.py；src/cspgnet/official_pibnet_edge_backend.py | 同一脚本 --method M3 |
| Geo-strong Erwin | 独立 Geo-strong 实验 | scripts/point_geo_strong_erwin.py；point_reference_models.py；third_party/Erwin/models/erwin.py | scripts/train_geo_strong_erwin_local.py |

M0/M4/M5 的组合式主干来自现有实验，未另写模型或改动参数。
更早的普通 TransolverGeoStrong 在 src/cspgnet/v3_transolver_representations.py，
不应与 M5CompositionalGeoStrong 混同。

- 共享几何编码器：src/cspgnet/v3_learning.py 的 GeoStrongAdapted；M5 使用其对应局部/全局结构。
- 原骨干源码：third_party/Transolver、third_party/Transolver_plus、third_party/Erwin。
- 预处理 / 评估：scripts/prepare_v6s11_20k_all_methods.py、evaluate_torus_20k_review.py、prepare_point_reference_tests.py。
- Windows Erwin C ABI：scripts/erwin_cabi.cpp、erwin_windows_binding.py；原作者 ball-tree 源码在 work/point_reference_packages_20261010/balltree_erwin-0.1.0。
- work 中仅保留必要的 timm 函数源码和 ball-tree C/Cython 源码，不是数据或安装缓存。

## 未迁移的数据和权重

以下路径相对于原实验工程根目录；本公开说明不包含个人电脑的绝对路径。

以下都是原工程下的相对位置，未复制到本目录：

- outputs/v6s11_20k_all_methods_20261009/prepared_retry5：20,000训练、2,000验证及部分测试缓存。
- outputs/v6s11_20k_all_methods_20261009/torus_review_20261010/cache：Torus测试缓存。
- outputs/point_references_20k_20261010/test_sidecar：N8及N8-OOD测试缓存。
- outputs/point_surface_20k_20261011/geometry_v3：真实点云和法向。
- outputs/erwin_geo_strong_local_20261013/geometry：Erwin nearest16 / tree。
- outputs/v6s11_20k_all_methods_20261009/remote_results/final_20261010：M0/M3/M4/M5权重及结果。
- outputs/erwin_geo_strong_local_20261013/runs/erwin：Erwin权重及六测试结果。
- datasets/canonical_response_v3：原始数据。

## 使用边界与复现准备

这是代码迁移，不是自带数据和环境的“一键运行包”。
模型、训练、损失和预处理文件保持原源码字节不变；CODE_MANIFEST.json 记录 SHA-256 与来源。
运行前需安装依赖并配置数据路径、官方依赖和 Windows ball-tree 动态库。
point_reference_runtime.py / official_pibnet_train.py 中保留了历史本机路径；
Erwin 默认还按原相对目录读取数据，不能在缺少这些文件时直接启动训练。

M0/M3/M4/M5 原入口支持 --cache 指向现有 prepared_retry5，--run-root 指向新的训练输出。
先配置环境后查看其 --help；不要将 --run-root 设为原参考实验目录。
Erwin 的源绑定、资格检查与本机执行顺序见 ERWIN_GEO_STRONG_LOCAL_20261013.md；
其他原始配置见同目录的实验协议文件。编译/安装产物不随本代码包迁移。

requirements_model_runtime.txt 是依据已有实验环境整理的核心依赖参考，
original_requirements/requirements.txt 是原仓库要求；两者不是已验证的全平台锁文件。
本次未安装新环境、重新训练或重新执行测试。

只检查源码完整性及 Python 语法（不加载 Torch、不训练）：

```text
python inspect_sources.py
```

## 实验解释

M0/M3/M4/M5 原辅助损失 λq=0.1；本地 Geo-strong Erwin 为 λq=0 的仅场损失。
原实验是 seed0、200轮、batch16；输入/解码器/参数量及部分损失不同，不能称等参数纯输入消融。
标签仍有探索性及正式 BEM/截断门槛未补齐的限制，不能因代码迁移宣称全部物理审核已通过。
第三方原许可证随源码保留。
