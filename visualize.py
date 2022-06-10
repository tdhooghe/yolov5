# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from pathlib import Path


def read_file(path, fields=None):
    if os.path.splitext(path)[-1] == '.csv':
        df = pd.read_csv(path, usecols=fields)
    elif os.path.splitext(path)[-1] == '.xls' or '.xlsx':
        df = pd.read_excel(path, engine='openpyxl', usecols=fields)
    else:
        return print('Not a valid file')
    return df


# %%
# plots different scatter plots
def col_plot_speed_det(path):
    df = read_file(path)

    g = sns.relplot(
        data=df, x='FPS', y='mAP50', col='device', hue='model', style='precision', kind='scatter'

    )

    g.set_axis_labels("Inference speed in FPS", "Detection performance in mAP:@.5")
    g.set_titles(col_template="Device: {col_name}")
    # g.set(xlim=(0, 60), ylim=(0, 12), xticks=[10, 30, 50], yticks=[2, 6, 10])
    g.tight_layout()
    # g.savefig("results/plots/relplot.png")
    plt.show()


def plot_exp1_results(path):
    fields = ['model', 'precision', 'FPS', 'mAP50']
    df = read_file(path, fields)

    pivot = df.pivot(index='model', columns='precision', values=['FPS', 'mAP50'])

    # calculate percentage of mAP and FPS of INT8 and FP 16 compared to baseline FP32
    pivot.loc[:, ('FPS', 'fp16')] = (pivot['FPS']['fp16'] / pivot['FPS']['fp32'][:]) * 100
    pivot.loc[:, ('FPS', 'int8')] = (pivot['FPS']['int8'] / pivot['FPS']['fp32'][:]) * 100
    pivot.loc[:, ('FPS', 'fp32')] = 100
    pivot.loc[:, ('mAP50', 'fp16')] = (pivot['mAP50']['fp16'] / pivot['mAP50']['fp32'][:]) * 100
    pivot.loc[:, ('mAP50', 'int8')] = (pivot['mAP50']['int8'] / pivot['mAP50']['fp32'][:]) * 100
    pivot.loc[:, ('mAP50', 'fp32')] = 100

    df = pivot.stack().reset_index()

    sns.scatterplot(
        data=df, x='mAP50', y='FPS', hue='model', style='precision'
    )

    # g.set_axis_labels("Inference speed in FPS", "Detection performance in mAP:@.5")
    # g.set(xlim=(0, 60), ylim=(0, 12), xticks=[10, 30, 50], yticks=[2, 6, 10])
    # g.tight_layout()
    # g.savefig("results/plots/relplot.png")
    plt.show()


# %%
plot_exp1_results("./results/experiments/exp1/220530_exp1_1.xlsx")
