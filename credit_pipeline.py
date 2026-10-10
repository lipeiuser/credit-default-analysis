# -*- coding: utf-8 -*-
"""
个人信贷违约风险预测(信用评分方向) —— 项目代码骨架
================================================================
数据: Kaggle "Give Me Some Credit" (15万条样本, 11列)
  下载页: https://www.kaggle.com/competitions/GiveMeSomeCredit/data
  或命令行: kaggle competitions download -c GiveMeSomeCredit
  解压后将 cs-training.csv 放入本文件同级的 data/ 目录
  (如Kaggle访问不畅, 可在天池/和鲸数据集区搜索 "Give Me Some Credit" 的转载版)

依赖: pip install pandas numpy scipy scikit-learn lightgbm imbalanced-learn shap matplotlib

使用说明:
  这是一份"可跑通基线 + TODO改进点"的骨架代码。
  所有标 [TODO] 的位置是你需要自己查资料、做选择、补代码的地方——
  这些决定 + 数字, 正是面试时讲"个人贡献"的素材。
  每个阶段对应简历的一条要点, 跑完后把 TODO 结果里的数字填进简历模板。

字段说明 (11列):
  SeriousDlqin2yrs                      目标变量: 未来2年内是否90天以上逾期 (1=违约, 占比约6.7%)
  RevolvingUtilizationOfUnsecuredLines  循环额度使用率(已用/总授信)
  age                                   年龄
  NumberOfTime30-59DaysPastDueNotWorse  30-59天逾期次数
  DebtRatio                             月还款额/月收入
  MonthlyIncome                         月收入(约20%缺失)
  NumberOfOpenCreditLinesAndLoans       开放信用额度与贷款数
  NumberOfTimes90DaysLate               90天以上逾期次数
  NumberRealEstateLoansOrLines          房地产贷款数
  NumberOfTime60-89DaysPastDueNotWorse  60-89天逾期次数
  NumberOfDependents                    家庭被抚养人数(约2.6%缺失)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from scipy.stats import ks_2samp

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, recall_score

plt.rcParams['font.sans-serif'] = ['SimHei']      # Windows 中文字体
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'cs-training.csv')
FIG_DIR = os.path.join(BASE_DIR, 'figs')
TARGET = 'SeriousDlqin2yrs'
SEED = 42

PAST_DUE_COLS = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTime60-89DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
]


# ====================================================================
# 阶段1 读取与EDA —— 简历要点: "完成数据探索, 量化违约率/缺失率/异常值"
# ====================================================================
def load_data() -> pd.DataFrame:
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH, index_col=0)   # 第一列是无名行号
    print(f'数据形状: {df.shape}')
    return df


def eda(df: pd.DataFrame) -> dict:
    """EDA 并返回需要写进简历的基础数字。"""
    stats = {}
    # 1.1 标签分布(类别不平衡的量化证据)
    rate = df[TARGET].mean()
    stats['违约率'] = rate
    print(f'违约率: {rate:.2%}  (正样本 {int(df[TARGET].sum())} / {len(df)})')

    # 1.2 缺失率
    na = df.isnull().mean().sort_values(ascending=False)
    print('\n缺失率TOP5:\n', na.head())

    # 1.3 异常值线索: 用 describe 观察极端值
    #     预期发现: 额度使用率>>1、DebtRatio 极大、逾期次数出现 96/98、age 最小值很小
    print('\n数值特征 describe:\n', df.describe().T[['min', '50%', 'max']])

    # [TODO 1.4] 画3张图保存到 figs/, 面试讲项目时直接用:
    #   ① 额度使用率按违约分组的分布(kde/boxplot)
    #   ② 三类逾期次数的分组违约率柱状图(次数分箱后)
    #   ③ MonthlyIncome 缺失 vs 未缺失 两组的违约率对比(缺失是否有信息量)
    # 提示: df.groupby(分箱列)[TARGET].mean() 即可得到分组违约率
    return stats


# ====================================================================
# 阶段2 数据清洗
# ====================================================================
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 2.1 额度使用率: 超过1的属于"刷爆/异常", 截断到1
    #     [TODO 进阶] 也可以不截断, 改为单独生成 use_over_limit 二值特征, 对比哪种AUC高
    cap_col = 'RevolvingUtilizationOfUnsecuredLines'
    df[cap_col] = df[cap_col].clip(upper=1)

    # 2.2 逾期次数中的 96/98 是"特殊编码"(表示多次逾期), 不是真实次数
    #     处理: 生成极端标记列, 原值置为缺失再填充 —— [TODO] 你来决定填充策略(中位数/0/分组中位数)
    for col in PAST_DUE_COLS:
        df[col + '_extreme'] = (df[col] >= 96).astype(int)
        df.loc[df[col] >= 96, col] = np.nan
        # 基线默认填0: 非极端样本的中位数本身就是0, "多次逾期"的信息已由 _extreme 列保留
        # [TODO 进阶] 对比: 填0 vs 填中位数 vs 保留96/98原值不置NaN, 看下游AUC差异并记录结论
        df[col] = df[col].fillna(0)

    # 2.3 MonthlyIncome 缺失约20%: 中位数填充 + 缺失标记列
    #     [TODO] 对比: 中位数填充 vs 按 age 分组中位数填充, 哪种下游AUC更高
    df['MonthlyIncome_na'] = df['MonthlyIncome'].isnull().astype(int)
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())

    # 2.4 NumberOfDependents 缺失约2.6%: 填0(无被抚养人是最常见取值)
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)

    # 2.5 [TODO] age=0 或极端 DebtRatio 的处理方式, 记录你的选择和理由
    return df


# ====================================================================
# 阶段3 特征工程 —— 简历要点: "构造交叉特征, 通过IV筛选强预测特征"
# ====================================================================
def calc_iv(df: pd.DataFrame, feature: str, bins: int = 10) -> float:
    """数值特征等频分箱后计算 IV 值(风控特征筛选标准指标, 面试常问)。"""
    x = pd.qcut(df[feature].rank(method='first'), q=bins, duplicates='drop')
    grouped = df.groupby(x, observed=True)[TARGET].agg(['sum', 'count'])
    grouped['bad_rate'] = grouped['sum'] / grouped['sum'].sum()
    grouped['good_rate'] = (grouped['count'] - grouped['sum']) / (grouped['count'].sum() - grouped['sum'].sum())
    grouped = grouped[(grouped['bad_rate'] > 0) & (grouped['good_rate'] > 0)]
    return float(((grouped['bad_rate'] - grouped['good_rate']) *
                  np.log(grouped['bad_rate'] / grouped['good_rate'])).sum())


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 3.1 交叉特征(业务逻辑: 偿还压力/依赖负担)
    df['income_per_dependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
    df['debt_per_line'] = df['DebtRatio'] / (df['NumberOfOpenCreditLinesAndLoans'] + 1)
    df['total_past_due'] = df[PAST_DUE_COLS].sum(axis=1)

    # 3.2 偏态变量对数变换
    df['log_DebtRatio'] = np.log1p(df['DebtRatio'].clip(lower=0))
    df['log_income'] = np.log1p(df['MonthlyIncome'])

    # 3.3 [TODO 核心] IV 筛选: 对所有候选特征调用 calc_iv,
    #     按 IV 排序输出表格, 保留 IV>0.02 的特征(经验阈值),
    #     把 "Top5 特征及IV值" 记下来 -> 简历+面试素材
    ivs = {c: calc_iv(df, c) for c in df.drop(columns=[TARGET]).select_dtypes(include=np.number).columns}
    iv_series = pd.Series(ivs).sort_values(ascending=False)
    print('\n特征IV值排序:\n', iv_series.round(3))
    return df


# ====================================================================
# 阶段4 不平衡处理 + 模型对比 —— 简历要点: "对比类别加权与SMOTE"
# ====================================================================
def get_models():
    return {
        'LR': LogisticRegression(max_iter=2000, random_state=SEED),
        'RF': RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
        'LGB': lgb.LGBMClassifier(n_estimators=300, n_jobs=-1, random_state=SEED, verbose=-1),
    }


def compare_imbalance(X_tr, y_tr, X_val, y_val):
    """策略A: class_weight;  策略B: SMOTE(只作用于训练集! 验证集必须保持原始分布, 面试考点)."""
    rows = []
    for name, model in get_models().items():
        # 策略A
        m_a = model
        if hasattr(model, 'class_weight'):
            m_a.set_params(class_weight='balanced')
        m_a.fit(X_tr, y_tr)
        prob_a = m_a.predict_proba(X_val)[:, 1]
        rows.append([name + '+加权', roc_auc_score(y_val, prob_a),
                     recall_score(y_val, (prob_a >= 0.5).astype(int))])

        # 策略B: SMOTE
        try:
            from imblearn.over_sampling import SMOTE
            X_res, y_res = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
            m_b = model.__class__(**{k: v for k, v in model.get_params().items()})
            m_b.fit(X_res, y_res)
            prob_b = m_b.predict_proba(X_val)[:, 1]
            rows.append([name + '+SMOTE', roc_auc_score(y_val, prob_b),
                         recall_score(y_val, (prob_b >= 0.5).astype(int))])
        except ImportError:
            print('提示: pip install imbalanced-learn 后再对比SMOTE')

    result = pd.DataFrame(rows, columns=['模型+策略', 'AUC', '召回率']).sort_values('AUC', ascending=False)
    print('\n不平衡处理策略对比:\n', result.round(4))
    # [TODO] 加上 LightGBM(需要 pip install lightgbm), 它通常是最优模型
    return result


# ====================================================================
# 阶段5 评估: AUC + KS —— KS 是风控必问指标
# ====================================================================
def ks_score(y_true, y_prob) -> float:
    """KS = 好坏样本分数分布的最大间隔。"""
    return float(ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0]).statistic)


# ====================================================================
# 阶段6 调参 —— 简历要点: "调参后AUC从x提升到y"
# ====================================================================
def tune_best_model(X_tr, y_tr):
    """对 LGB+加权 做网格调参, 用5折交叉验证AUC均值作为评估(更严谨)."""
    from sklearn.model_selection import GridSearchCV

    base = lgb.LGBMClassifier(class_weight='balanced', random_state=SEED, verbose=-1, n_jobs=-1)
    param_grid = {
        'n_estimators': [300, 500],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63],
        'max_depth': [-1, 6],
    }
    gs = GridSearchCV(
        base, param_grid, scoring='roc_auc', cv=5, n_jobs=1, verbose=1
    )
    gs.fit(X_tr, y_tr)
    print(f'\n调参最优参数: {gs.best_params_}')
    print(f'调参后5折CV AUC: {gs.best_score_:.4f}')
    print('对照: 调参前 LGB+加权 单次划分 AUC ≈ 0.8600')
    return gs.best_estimator_


# ====================================================================
# 阶段7 SHAP可解释 —— 简历要点: "输出Top5违约驱动因子"
# ====================================================================
def explain(model, X_val: pd.DataFrame):
    import shap
    sample = X_val.sample(min(2000, len(X_val)), random_state=SEED)
    explainer = shap.TreeExplainer(model)
    # shap >= 0.44 新API: explainer(sample) 返回 Explanation 对象
    explanation = explainer(sample)
    shap_values = explanation.values
    # 二分类时 values 可能是 (n, 2), 取第1列(正类)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    # 输出 Top5 违约驱动因子(平均|SHAP|)
    mean_abs = np.abs(shap_values).mean(axis=0)
    top5 = pd.Series(mean_abs, index=sample.columns).sort_values(ascending=False).head(5)
    print('\nTop5 违约驱动因子(平均|SHAP|):\n', top5.round(4))

    shap.summary_plot(shap_values, sample, show=False)
    plt.savefig(os.path.join(FIG_DIR, 'shap_summary.png'), bbox_inches='tight', dpi=150)
    print('SHAP图已保存到 figs/shap_summary.png')
    return top5


# ====================================================================
# 阶段8 风险分层与业务建议 —— 把"建模"升级为"业务分析"的亮点
# ====================================================================
def risk_stratification(y_val, y_prob, n_bins: int = 10):
    """按预测违约概率等频分箱, 输出各档实际违约率 -> 佐证模型排序能力并给出授信建议。"""
    df = pd.DataFrame({'prob': y_prob, 'label': y_val})
    df['bin'] = pd.qcut(df['prob'], q=n_bins, duplicates='drop')
    table = df.groupby('bin', observed=True).agg(
        用户数=('label', 'size'), 实际违约率=('label', 'mean')).reset_index()
    print('\n按预测概率分箱的校准表(预期: 概率越高档实际违约率越高):\n', table.round(4))

    # 按预测概率分位数归并为低/中/高三档(60%/30%/10%)
    df['tier'] = pd.qcut(df['prob'], q=[0, 0.6, 0.9, 1.0], labels=['低风险', '中风险', '高风险'])
    tier_table = df.groupby('tier', observed=True).agg(
        用户占比=('label', lambda s: len(s) / len(df)),
        实际违约率=('label', 'mean')).reset_index()
    print('\n三档风险分层:\n', tier_table.round(4))
    print('\n业务建议:')
    for _, row in tier_table.iterrows():
        if row['tier'] == '高风险':
            print(f"  高风险档(违约率{row['实际违约率']:.1%}, 占比{row['用户占比']:.0%}): 建议拒贷或降额")
        elif row['tier'] == '中风险':
            print(f"  中风险档(违约率{row['实际违约率']:.1%}, 占比{row['用户占比']:.0%}): 建议人工审核")
        else:
            print(f"  低风险档(违约率{row['实际违约率']:.1%}, 占比{row['用户占比']:.0%}): 建议快速审批通道")
    return table, tier_table


# ====================================================================
# 主流程
# ====================================================================
def main():
    df = load_data()
    eda(df)
    df = clean(df)
    df = feature_engineering(df)

    # 划分训练/验证集: 分层抽样保持违约率一致
    y = df[TARGET]
    X = df.drop(columns=[TARGET]).select_dtypes(include=np.number)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25,
                                                stratify=y, random_state=SEED)

    compare_imbalance(X_tr, y_tr, X_val, y_val)

    # 阶段6: 对最优模型 LGB+加权 调参
    best_model = tune_best_model(X_tr, y_tr)

    # 阶段5: 计算 KS(风控必问指标)
    y_prob = best_model.predict_proba(X_val)[:, 1]
    ks = ks_score(y_val, y_prob)
    auc = roc_auc_score(y_val, y_prob)
    print(f'\n调参后验证集 AUC={auc:.4f}, KS={ks:.4f}')

    # 阶段7: SHAP 可解释性
    explain(best_model, X_val)

    # 阶段8: 风险分层与业务建议
    risk_stratification(y_val, y_prob)

    print('\n全部流程跑完! 把上面的数字填进数字记录表, 即可开始写简历。')


if __name__ == '__main__':
    main()
