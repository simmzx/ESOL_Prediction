# ESOL LogS Prediction

## 0. 设计理由

模型两条路线：

- **RNN 主线** PyTorch 复现：Embedding(64) → BiLSTM(64) → BiLSTM(32) → Dense(64) → Dropout(0.3) → logS；及改进版 Embedding(64) → BiGRU(64) → masked additive attention → Dense(64) → logS。
- **高性能小样本路线**：从 SMILES 计算 RDKit 7/28维描述符 + Morgan(半径2、1024位)；仅在训练集拟合缺失值处理器；候选包括 Ridge、RF、ExtraTrees、SVR、XGBoost；比较 11 种预设模型组合，并以 **验证集RMSE最小**的非 Dummy 模型作为选定模型。XGBoost 默认使用CPU（仅1,128行，CPU histogram 往往更快），可选GPU。**不能因为测试集上另一个模型更好而更换模型。**

致谢本项目借鉴的开源项目：
- https://github.com/Skurau/Delaney-ESOL （RF/GNN及分析框架）
- https://github.com/wandixu/esol-solubility-prediction （RDKit、Morgan、XGB、scaffold split）
- https://github.com/chemprop/chemprop （MPNN延伸方向；出于速度和小样本稳定性考虑，主实验不依赖Chemprop）
- https://github.com/githublearner123456/esol （描述符+指纹DNN参考）

## 1. 数据情况（已通过代码检查）

位置： `data/esol.csv`

- 共1128行、10列；无空值、无无效SMILES。
- target是 `measured log solubility in mols per litre`，范围 `[-11.60, 1.58]`。
- `ESOL predicted log solubility in mols per litre` 属于原数据自带外部预测，**不得纳入特征或训练标签**。
- Canonical SMILES有1117种，11组重复结构，6组有测量标签冲突；最大差值1.03。为了保留实测数据，本项目**不删除、也不擅自平均冲突测量**，而使用 canonical-SMILES 分组，避免同一结构跨拆分。
- Murcko scaffold共269组，其中空骨架（无环）317条；scaffold拆分将整个空骨架视为同组，严格但分布存在偏差。

主实验随机分组种子42，约72%训练/8%验证/20%测试；真实分配为**811/90/227**。
scaffold补充实验按独立骨架分组，分配为**813/89/226**。注意这种分布不等同其他论文的scaffold split。**Random与Scaffold分数不能互相当作同一基准**。

## 2. 服务器部署

```bash
conda create -n esol_lab2 python=3.11 -y
conda activate esol_lab2
python -m pip install -U pip
python -m pip install -r requirements.txt

# python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

CUDA_VISIBLE_DEVICES=1 python scripts/check_env.py --require-gpu
```

## 3. 运行

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

出现 `[SUCCESS]` 表示全流程成功。`nohup` 持久化后台任务

### 快速首跑（单随机种子）

```bash
bash scripts/launch_background.sh --data /home/zhangxiang_cluster/assign1/esol.csv --output /home/zhangxiang_cluster/assign1/esol_lab2_complete/results/quick_random --mode fast --gpu 1
```

### 骨架泛化附加实验

```bash
bash scripts/launch_background.sh --data /home/zhangxiang_cluster/assign1/esol.csv --output /home/zhangxiang_cluster/assign1/esol_lab2_complete/results/secondary_scaffold --split scaffold --mode fast --gpu 1
```

注意：骨架实验是单独完整训练，**不能拿random训练的模型直接在同一个random-test上伪称scaffold-test**。比较时报告划分细节。


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


## 5. 统计与实验规范

1. **公平**：ML与RNN共用同一个训练/验证/测试划分和logS标签；训练和目标标准化不能使用测试数据。分子重复结构不得跨集合。
2. **选择模型**：候选模型和超参数在 `src/train_ml.py` 中预注册，优选指标是验证集RMSE；最终只计算选中ML模型在测试集的结果、Dummy测试结果及预声明RNN模型的测试指标。
3. **稳定性**：主模式对RNN进行3种子重复，报告均值±样本标准差；ML仅一个数据分割和一个候选搜索，**不能将其单次成绩表达成多次训练的均值或科学确定的“最佳”**。
4. **交叉验证更强**：如需要完整机器学习论文级结论，可以在不查看测试集的前提下再采用训练部分的GroupKFold嵌套CV；这套课程作业优先保证流程可复现、时间可控。
5. **对参考项目的解释**：不同作者的测试集、去重、数据预处理不一致，网上0.55/0.80 RMSE不应与本项目测试分数直接排名。
6. **解释用于对照的RNN基线**：这是层结构与训练参数匹配的PyTorch implementation，非原TensorFlow/Keras复现；比如使用验证最好epoch进行检查点选择、固定分组拆分。因此写“teacher-inspired BiLSTM baseline”而非“exact original result”。
