# ESPC-Transolver-code

ESPC-Transolver 等效表面散射实验及几何对照方法的**仅源码发布**。

本仓库整理 ESPC-Transolver、Transolver-Geo-strong、Transolver-Geo-matched、PIBNet
和 Geo-strong Erwin 的模型、输入适配、训练/评估及预处理代码。
**不包含数据缓存、原始样本、模型权重、训练结果或 Python/CUDA 安装环境。**
这不是下载后无需配置即可训练的一键复现包。

## 方法与源码

下列参数量对应已有实验的记录配置，不代表等参数对照。

| 方法 | 标识 | 主要实现 | 参数量 | 辅助损失 λq |
|---|---|---|---:|---:|
| ESPC-Transolver | M0 | [组合式主干](src/cspgnet/v3_compositional_transolver.py)、[物体间模块](src/cspgnet/v3_object_transolver.py) | 2,076,765 | 0.1 |
| Transolver-Geo-strong | M5 | [M5CompositionalGeoStrong](scripts/train_v6s11_20k_all_methods.py) | 2,143,581 | 0.1 |
| Transolver-Geo-matched | M4 | 同一组合式主干；[PreparedScenes](scripts/train_v6s11_20k_all_methods.py) 中的几何占据输入 | 2,076,765 | 0.1 |
| PIBNet | M3 | [官方骨干](src/pibnet/gnn/pibnet.py)、[edge-index 适配](src/cspgnet/official_pibnet_edge_backend.py) | 5,220,930 | 0.1 |
| Geo-strong Erwin | 独立实验 | [几何适配器](scripts/point_geo_strong_erwin.py)、[规范节点模型](scripts/point_reference_models.py)、[Erwin 骨干](third_party/Erwin/models/erwin.py) | 3,753,466 | 0 |

主要输入区别：

- **M0 / ESPC-Transolver**：由既有散射响应构造的 S(Z) 特征，加已知入射与查询条件。
- **M4 / Geo-matched**：几何占据描述替换响应特征，保留对应主干及输入通道接口。
- **M5 / Geo-strong**：真实表面点与法向经局部/全局编码后送入组合式主干。
- **M3 / PIBNet**：响应特征经官方输入与图结构适配，采用已有 edge-index 执行后端。
- **Geo-strong Erwin**：每物体 2,048 个真实点及法向，经 nearest16 编码到每物体
  128 个规范节点；128 个几何通道与 12 个已知条件通道组成 140 通道输入，不加载 S(Z)/Z/q。

更早的普通 `TransolverGeoStrong` 位于
[v3_transolver_representations.py](src/cspgnet/v3_transolver_representations.py)，
**不等同于 M5CompositionalGeoStrong**，请勿混用模型标识或权重。

## 目录

```text
src/cspgnet/          等效表面模型、输入表示、损失及模型适配器
src/pibnet/           PIBNet 骨干及相关代码
scripts/              训练、评估、预处理及本机执行脚本
third_party/          原作者骨干源码及随附许可
work/                 必要 timm 函数及 ball-tree C/Cython 源码，不含安装缓存
original_requirements/ 原仓库依赖说明
CODE_MANIFEST.json    文件大小、SHA-256 与相对来源记录
inspect_sources.py    仅依赖 Python 标准库的源码完整性/语法检查
```

## 快速检查

源码检查无需安装 Torch，不会启动模型、访问数据或执行训练：

```bash
python -B inspect_sources.py
```

检查内容包括清单内文件的 SHA-256、Python 语法，以及不应出现的数据/权重/运行库文件。
`.gitattributes` 保留文件原始换行，避免 Git 自动转换换行导致源绑定哈希变化。
这项检查**不等同于数值精度、依赖环境或训练复现验证**。

## 配置依赖与数据

1. 根据 [核心依赖参考](requirements_model_runtime.txt) 和
   [原仓库依赖](original_requirements/requirements.txt) 配置独立环境。
   历史模型环境采用 Python 3.12、Torch 2.8.0 + CUDA 12.8；CUDA 扩展须与目标平台匹配。
   两份依赖说明都不是已验证的全平台锁文件。
2. 自行准备具有相同结构、统计量及身份清单的数据缓存。本仓库没有数据下载入口。
3. 配置脚本中的路径。尤其是 `point_reference_runtime.py`、`official_pibnet_train.py`
   保留了历史本机依赖路径；Erwin 还依赖原相对数据布局和 Windows ball-tree 构建产物。
4. 保留原有数据/源码身份检查，不要用占位数组绕过检查或覆盖参考实验输出。

### 原实验的数据协议（数据未发布）

| 部分 | 场景数 |
|---|---:|
| canonical-v3 / v6s11 训练 | 10,000 + 10,000 |
| 两组独立验证 | 1,000 + 1,000 |
| ID / N8 测试 | 1,000 / 1,000 |
| Shape OOD / Topology OOD | 500 / 500 |
| Shape-N8 OOD / Topology-N8 OOD | 250 / 250 |

原缓存相对位置及准备边界见 [中文代码说明](README_代码说明.md)。
未声明数据或权重已公开，也未提供论文正式数据的发布承诺。

## 训练与评估入口

完成环境和数据路径配置后，M0/M3/M4/M5 使用同一入口。例如：

```bash
python scripts/train_v6s11_20k_all_methods.py \
  --cache /path/to/prepared_retry5 \
  --run-root /path/to/new_runs \
  --method M0 --seed 0 --workers 0
```

将 `M0` 替换为 `M3`、`M4` 或 `M5` 可选择对应方法。
该入口的单方法模式不自动完成全部六测试；评估流程还需按原协议另行配置。
已有评估/预处理入口包括：

- [evaluate_torus_20k_review.py](scripts/evaluate_torus_20k_review.py)
- [prepare_v6s11_20k_all_methods.py](scripts/prepare_v6s11_20k_all_methods.py)
- [prepare_point_reference_tests.py](scripts/prepare_point_reference_tests.py)

Geo-strong Erwin 的独立入口为
[train_geo_strong_erwin_local.py](scripts/train_geo_strong_erwin_local.py)：

```bash
python scripts/train_geo_strong_erwin_local.py --qualify
python scripts/train_geo_strong_erwin_local.py --train
```

上面两条命令仅在完成其 Windows/CUDA 依赖、数据布局、动态库和源绑定配置后可用。
资格检查通过后才训练；请勿跳过资格或并行启动重复 worker。
详细协议见 [Geo-strong Erwin 实验记录](ERWIN_GEO_STRONG_LOCAL_20261013.md)
和 [mixed20k 实验补充协议](V6S11_20K_ALL_METHODS_RETRAIN_ADDENDUM_20261009.md)。

## 结果解释与限制

- 原记录配置为 seed0、200 轮、batch16，best checkpoint 仅由验证集选择。
- M0/M3/M4/M5 的辅助损失 λq=0.1；Geo-strong Erwin 为 λq=0 的仅场损失，无通量指标。
- 骨干、参数量、编码器/解码器及部分损失存在差异，不能称等参数或纯输入消融。
- 测试集已被查看；单种子结果属于探索性对照。
- 部分标签的正式 BEM/截断门槛尚未补齐，不能视为论文正式物理审核已全部通过。
- 本次仓库整理只检查源码，不重新执行 200 轮训练或模型测试。

## 第三方来源与许可

请保留原作者署名及各组件的许可文件。详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
本次整理不对全部代码另授一份统一许可证，也不扩张第三方代码的使用权限。
