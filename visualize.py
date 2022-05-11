# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# %%
def plot_speed_det(df_path):
    df = pd.read_csv(df_path)
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
