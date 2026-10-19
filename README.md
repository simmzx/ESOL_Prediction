# ESOL LogS Prediction

## 0. 目标及设计理由

本项目针对课程《Advanced Topics in Artificial Intelligence Drug Discovery》Lab 2（RNN）：使用指定 ESOL 数据集预测实测水溶解度 logS，按照 Data preparation → Data cleaning → Model construction → Training & Test → Plot results → Review & Optimize 形成独立报告。

模型两条路线：

- **RNN 主线（对应 Lab 2）**：课程结构的 PyTorch 复现：Embedding(64) → BiLSTM(64) → BiLSTM(32) → Dense(64) → Dropout(0.3) → logS；及改进版 Embedding(64) → BiGRU(64) → masked additive attention → Dense(64) → logS。使用 GPU:1。老师 PPT 使用 TensorFlow/Keras，**本代码忠实对应结构但不是完全同框架逐行复现**，有按分子分组划分、train-only词表、梯度截断及最佳验证检查点等差异，需要在报告中说明。
- **高性能小样本路线**：从 SMILES 计算 RDKit 7/28维描述符 + Morgan(半径2、1024位)；仅在训练集拟合缺失值处理器；候选包括 Ridge、RF、ExtraTrees、SVR、XGBoost；比较 11 种预设模型组合，并以 **验证集RMSE最小**的非 Dummy 模型作为选定模型。XGBoost 默认使用CPU（仅1,128行，CPU histogram 往往更快），可选GPU。**不能因为测试集上另一个模型更好而更换模型。**

本项目借鉴开源项目的一般实验思想而非复制其源码：
- https://github.com/Skurau/Delaney-ESOL （RF/GNN及分析框架）
- https://github.com/wandixu/esol-solubility-prediction （RDKit、Morgan、XGB、scaffold split）
- https://github.com/chemprop/chemprop （MPNN延伸方向；出于速度和小样本稳定性考虑，主实验不依赖Chemprop）
- https://github.com/githublearner123456/esol （描述符+指纹DNN参考；本实验不直接复制）

## 1. 你的CSV实际情况（已通过代码检查）

位置：`/home/zhangxiang_cluster/assign1/esol.csv`（此包额外放了一份用户提供的 `data/esol.csv`，便于离线测试）。

- 共1128行、10列；无空值、无无效SMILES。
- target是 `measured log solubility in mols per litre`，范围 `[-11.60, 1.58]`。
- `ESOL predicted log solubility in mols per litre` 属于原数据自带外部预测，**不得纳入特征或训练标签**。
- Canonical SMILES有1117种，11组重复结构，6组有测量标签冲突；最大差值1.03。为了保留实测数据，本项目**不删除、也不擅自平均冲突测量**，而使用 canonical-SMILES 分组，避免同一结构跨拆分。
- Murcko scaffold共269组，其中空骨架（无环）317条；scaffold拆分将整个空骨架视为同组，严格但分布存在偏差。

主实验随机分组种子42，约72%训练/8%验证/20%测试；真实分配为**811/90/227**。
可选scaffold补充实验按独立骨架分组，分配为**813/89/226**。注意这种分布不等同其他论文的scaffold split，需要清楚描述构造方法。**Random与Scaffold分数不能互相当作同一基准**。

## 2. 服务器部署

```bash
cd /home/zhangxiang_cluster/assign1
# 先把本zip上传到这里（示例文件名 esol_lab2_complete.zip）
unzip esol_lab2_complete.zip
cd esol_lab2_complete

# 查看GPU编号：确认编号1有空闲显存
nvidia-smi -L
nvidia-smi -i 1

# 建议 Python 3.11 环境。若已有正确的GPU PyTorch环境，也可直接使用。
conda create -n esol_lab2 python=3.11 -y
conda activate esol_lab2
python -m pip install -U pip
python -m pip install -r requirements.txt

# PyTorch GPU单独根据系统CUDA驱动安装，参考 https://pytorch.org/get-started/locally/
# 例如服务器兼容 PyTorch cu126 时（仅作为示例，需根据你的实际驱动选择）：
# python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

CUDA_VISIBLE_DEVICES=1 python scripts/check_env.py --require-gpu
```

如果你已安装了正确CUDA版PyTorch，可以省略它的安装步骤；如果 PyTorch 显示 `+cpu` 或CUDA不可用，运行器默认停止，不会伪称使用GPU1。容器/Slurm中的GPU编号可能被重新映射，应以 `nvidia-smi -L` 和 CUDA 可见性为准。

## 3. 一条命令挂后台（推荐）

从项目根目录执行：

```bash
bash scripts/launch_background.sh \
  --data /home/zhangxiang_cluster/assign1/esol.csv \
  --output /home/zhangxiang_cluster/assign1/esol_lab2_complete/results/primary_random \
  --split random --seed 42 --mode standard --gpu 1 --threads 4
```

脚本立即打印 PID、日志路径，实验在 `nohup` 后台执行。标准模式对两种RNN分别运行 **42/43/44 共3个随机种子**；机器学习验证搜索只跑一次且配置固定。

查看：

```bash
tail -f /home/zhangxiang_cluster/assign1/esol_lab2_complete/results/primary_random/nohup.log
cat /home/zhangxiang_cluster/assign1/esol_lab2_complete/results/primary_random/launcher.pid
ps -fp "$(cat /home/zhangxiang_cluster/assign1/esol_lab2_complete/results/primary_random/launcher.pid)"
nvidia-smi -i 1
```

出现 `[SUCCESS]` 表示全流程成功。**不要并行启动两个GPU实验而无视显存容量**。`nohup` 持久化后台任务，管理员终止节点或作业调度清理进程不受其保护；集群如果要求Slurm，应该改用 `sbatch` 正规调度，不能越过队列直接占用GPU。

### 快速首跑（单随机种子）

```bash
bash scripts/launch_background.sh --data /home/zhangxiang_cluster/assign1/esol.csv --output /home/zhangxiang_cluster/assign1/esol_lab2_complete/results/quick_random --mode fast --gpu 1
```

### 骨架泛化附加实验（可选）

```bash
bash scripts/launch_background.sh --data /home/zhangxiang_cluster/assign1/esol.csv --output /home/zhangxiang_cluster/assign1/esol_lab2_complete/results/secondary_scaffold --split scaffold --mode fast --gpu 1
```

注意：骨架实验是单独完整训练，**不能拿random训练的模型直接在同一个random-test上伪称scaffold-test**。比较时报告划分细节。

### 完全无GPU情况下的诊断

```bash
bash scripts/run_all.sh --data data/esol.csv --output results/cpu_diagnostic --allow-cpu --mode fast
```

在CPU上也能跑，但BiLSTM/GRU训练时间可能增加。GPU1推荐。

## 4. 输出文件与PDF六阶段映射

| Workflow | 实验产物 | 解释 |
|---|---|---|
| Data preparation | `audit.json`, `distribution.png`, `split_membership.csv` | 样本信息、目标分布、分割 |
| Data cleaning | `duplicate_structures.csv`, `excluded_invalid.csv`, `clean_data.csv` | 重复结构/冲突标签、过滤数量 |
| Model construction | `src/core.py`, `src/train_ml.py`, `src/train_rnn.py` | 特征、ML候选、BiLSTM、BiGRU attention |
| Training & Test | `ml/validation_leaderboard.csv`, `ml/test_metrics.json`, `rnn/all_seed_metrics.csv` | 验证/测试，均在原始logS尺度 |
| Plot the results | `ml/validation_comparison.png`, `ml/test_scatter_residual.png`, `rnn/*/seed_*/learning_curve.png` | 对比、残差、学习曲线 |
| Review & Optimize | `ml/feature_importance.csv`, `rnn/seed_summary.csv`, `REPORT_SUMMARY.md` | 特征贡献、不同随机种子稳定性、模型总结 |

模型文件：`ml/selected_model.joblib`、`rnn/*/seed_*/checkpoint.pt`。测试集逐分子预测：`ml/test_predictions.csv`、`rnn/*/seed_*/test_predictions.csv`。

`REPORT_SUMMARY.md` 自动填充真实成绩，仅作为报告写作底稿；最终PDF仍应撰写实验目的/文献依据/实验方法/必要代码/结果图注/局限性，不能只交摘要。

## 5. 统计与实验规范

1. **公平**：ML与RNN共用同一个训练/验证/测试划分和logS标签；训练和目标标准化不能使用测试数据。分子重复结构不得跨集合。
2. **选择模型**：候选模型和超参数在 `src/train_ml.py` 中预注册，优选指标是验证集RMSE；最终只计算选中ML模型在测试集的结果、Dummy测试结果及预声明RNN模型的测试指标。
3. **稳定性**：主模式对RNN进行3种子重复，报告均值±样本标准差；ML仅一个数据分割和一个候选搜索，**不能将其单次成绩表达成多次训练的均值或科学确定的“最佳”**。
4. **交叉验证更强**：如需要完整机器学习论文级结论，可以在不查看测试集的前提下再采用训练部分的GroupKFold嵌套CV；这套课程作业优先保证流程可复现、时间可控。
5. **对参考项目的解释**：不同作者的测试集、去重、数据预处理不一致，网上0.55/0.80 RMSE不应与本项目测试分数直接排名。
6. **解释老师基线**：这是层结构与训练参数匹配的PyTorch implementation，非原TensorFlow/Keras复现；比如使用验证最好epoch进行检查点选择、固定分组拆分。因此写“teacher-inspired BiLSTM baseline”而非“exact original result”。

## 6. 实测结果（本地只对ML进行了完整正式训练）

在上传数据上：Random grouped 811/90/227，候选集不根据测试分数调整，**验证赢家为 SVR + 28个 RDKit 描述符**：

- Validation RMSE: **0.4623**
- Test RMSE: **0.6127**
- Test MAE: **0.4228**
- Test R²: **0.9104**
- Dummy Test RMSE: **2.0464**

这是一组具体的随机种子42测试结果，并非论文通用基准。数据、环境及库版本变化可能导致细微变化。深度模型的完整30/80 epoch、三种子GPU实验尚需在你的服务器上执行，不能提前编造成绩。

## 7. 预计耗时（估计，不是保证）

| 步骤 | 中等CPU + GPU1预估 |
|---|---:|
| CSV/RDKit/分割/图 | 5–30秒 |
| 11候选ML | 20秒–3分钟 |
| 教师BiLSTM 30 epochs ×1 seed | 20秒–3分钟 |
| BiGRU Attention 80 epochs上限 ×1 seed | 30秒–5分钟 |
| fast全流程 | 约2–12分钟 |
| standard全流程（两RNN×3 seed） | 约5–30分钟 |
| scaffold fast附加 | 约2–12分钟 |

时间随 GPU型号、共享节点负载、驱动、CPU线程、SMILES平均长度变化；首次环境安装、下载不计。上限80通常Early Stopping会提前停止。

**预估性能区间（风险较高的规划值，不是实验结果）**：random grouped ML RMSE 大概0.55–0.95，scaffold RMSE 0.85–1.6，RNN random RMSE 0.8–1.6，可能更差。严格划分/数据分布会显著改变结果；真实成绩以输出为准。

## 8. 最小检查测试

```bash
# 如果安装pytest： python -m pip install pytest
ESOL_CSV=/home/zhangxiang_cluster/assign1/esol.csv python -m pytest tests -q
bash -n scripts/run_all.sh scripts/launch_background.sh
python -m compileall -q src scripts
```

## 9. 建议报告标题与方法引用

**A Reproducible Study of Molecular Aqueous Solubility Prediction on ESOL: Comparing SMILES-based Recurrent Neural Networks and Physicochemical Descriptor Models**

原始ESOL: Delaney (2004), *ESOL: A Simple Model for Aqueous Solubility Based on Molecular Properties*, DOI: 10.1021/ci034243x。MoleculeNet: Wu et al. (2018), *MoleculeNet: A Benchmark for Molecular Machine Learning*, DOI: 10.1039/C7SC02664A。需在报告中引用RDKit、XGBoost、支持向量回归、BiLSTM/Attention相关资料。

## 10. 预测未见过的新SMILES（训练完成后）

```bash
# 传统ML：单个/多个SMILES
python -m src.predict --ml-model results/primary_random/ml/selected_model.joblib --smiles 'CCO' 'c1ccccc1' --output results/primary_random/new_molecules_ml.csv

# 也可以批量读取带 smiles 列的CSV
python -m src.predict --ml-model results/primary_random/ml/selected_model.joblib --csv new_molecules.csv --output new_molecules_predictions.csv

# 任一训练好的RNN检查点
python -m src.predict --rnn-checkpoint results/primary_random/rnn/bigru_attention/seed_42/checkpoint.pt --smiles 'CCO' 'c1ccccc1' --output new_molecules_rnn.csv
```

注意：对训练分布之外分子的预测可能不可靠；该接口不提供预测区间，也不应把 logS预测当作实测值。

### 本地另外完成的scaffold单次验证

使用相同程序但 scaffold 分组拆分（813/89/226）时，验证集选出 XGB_combined_smooth（28维+1024位），测试 RMSE=**0.9306**，MAE=**0.6968**，R²=**0.7824**。这与random结果属于不同测试分布，不可直接作为相同测试集的模型排名。
