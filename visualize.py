# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


# %%
def plot_speed_det(path):
    if os.path.splitext(path)[-1] == '.csv':
        df = pd.read_csv(path)
    elif os.path.splitext(path)[-1] == '.xls' or '.xlsx':
        df = pd.read_excel(path, engine='openpyxl')
    else:
        return print('Not a valid file')
    g = sns.relplot(
        data=df, x='FPS', y='mAP50', col='device', hue='model', style='precision', kind='scatter'

    )

    g.set_axis_labels("Inference speed in FPS", "Detection performance in mAP:@.5")
    g.set_titles(col_template="Device: {col_name}")
    # g.set(xlim=(0, 60), ylim=(0, 12), xticks=[10, 30, 50], yticks=[2, 6, 10])
    g.tight_layout()
    # g.savefig("results/plots/relplot.png")
    plt.show()


# %%
plot_speed_det("./results/experiments/exp2/220530_results_res-fps.xlsx")
