# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('results\plots\exp1_exp2_plot_data.csv')
#%%
print(df[])

#%%
# plt.figure(figsize=(8, 5))
g = sns.relplot(
    data=df, x='FPS', y='mAP50', col='device', hue='model', style='precision',kind='scatter'

)

g.set_axis_labels("Inference speed in FPS", "Detection performance in mAP:@.5")
g.set_titles(col_template="Device: {col_name}")
# g.set(xlim=(0, 60), ylim=(0, 12), xticks=[10, 30, 50], yticks=[2, 6, 10])
g.tight_layout()
g.savefig("results/plots/relplot.png")
plt.show()

#%%

