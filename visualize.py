# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
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
    # pivot.loc[:, ('FPS', 'fp16')] = (pivot['FPS']['fp16'] / pivot['FPS']['fp32'][:])
    # pivot.loc[:, ('FPS', 'int8')] = (pivot['FPS']['int8'] / pivot['FPS']['fp32'][:])
    # pivot.loc[:, ('FPS', 'fp32')] = 1
    # pivot.loc[:, ('mAP50', 'fp16')] = (pivot['mAP50']['fp16'] / pivot['mAP50']['fp32'][:])
    # pivot.loc[:, ('mAP50', 'int8')] = (pivot['mAP50']['int8'] / pivot['mAP50']['fp32'][:])
    # pivot.loc[:, ('mAP50', 'fp32')] = 1

    df = pivot.stack().reset_index()
    df['model_type'] = df['model'].apply(lambda x: 'p6' if '6' in x else 'p5')
    df['model'] = df['model'].apply(lambda x: x.replace('6', ''))
    cols = ['model', 'model_type', 'precision']
    df['code'] = df[cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    df['code'] = df['code'].str.replace("yolov5", "")
    #df = df.drop(df[df['precision'].str.contains('fp32')].index)
    plt.figure(figsize=(8, 5))

    # splot = sns.scatterplot(
    #     data=df, x='FPS', y='mAP50', size='model_type', size_order=['p6', 'p5'], hue='model', style='precision',
    #     palette='colorblind'
    # )
    splot = sns.scatterplot(
        data=df, x='FPS', y='mAP50', size='model_type', size_order=['p6', 'p5'], style='precision', hue='model',
        palette='colorblind'
    )
    #
    # splot = sns.scatterplot(
    #     data=df, x='FPS', y='mAP50', hue='model',
    #     palette='colorblind'
    # )

    # splot.set(xlabel='FPS/baseline FPS', ylabel='mAP50/baseline mAP50',
    #           #yscale='log'
    #           )

    # for i in range(df.shape[0]):
    #     plt.text(x=df.FPS[i] + 0.01, y=df.mAP50[i] + 0.005, s=df.code[i],
    #              fontdict=dict(color='black', size=10),
    #              # bbox=dict(facecolor='yellow', alpha=0.5)
    #              )

    plt.tight_layout()
    plt.show()
    # g.set_axis_labels("Inference speed in FPS", "Detection performance in mAP:@.5")
    # g.set(xlim=(0, 60), ylim=(0, 12), xticks=[10, 30, 50], yticks=[2, 6, 10])
    # g.tight_layout()
    # g.savefig("results/plots/relplot.png")
    # plt.legend(bbox_to_anchor=(1.01, 1.00), loc=2, borderaxespad=0.)
    # plt.tight_layout()
    # df = df.drop(df[df['precision'].str.contains('int8')].index)
    #
    # sns.scatterplot(
    #     data=df, x='FPS', y='mAP50', size='model_type', size_order=['p6', 'p5'], hue='model', style='precision',
    #     palette='colorblind'
    # )
    # plt.legend(bbox_to_anchor=(1.01, 1.00), loc=2, borderaxespad=0.)
    # plt.tight_layout()
    # plt.show()


# %%
plot_exp1_results("./results/experiments/exp1/220607_exp1_1.xlsx")
# %%
# df = read_file("./results/experiments/exp1/220530_exp1_1.xlsx")
# df['model_type'] = df['model'].apply(lambda x: 'p6' if '6' in x else 'p5')
# df['model'] = df['model'].apply(lambda x: x.replace('6', ''))
# print(df[df['model'].str.contains('yolov5n') & df['precision'].str.contains('int8')].index)
# #df_test = df_test.drop(df_test[df_test['model' == 'yolov5n'] & df_test['model_type'] == 'int8'].index)
