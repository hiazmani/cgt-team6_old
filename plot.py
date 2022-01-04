import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("results/largest_learning_rate/trappedBox12k copy.csv")

episodes = df['Episodes'].to_numpy()
rewards = df['Rewards'].to_numpy()
print(episodes)
# rewards = df.y
x_as = []
shittoplot = []
n_episodes = len(episodes)
div = n_episodes // 50
for step in range(div):
    curr_av = 0
    for x in range(50):
        curr_av += rewards[(step*50)+x]
    x_as.append(step * 50)
    shittoplot.append(curr_av / 50)     
    
plt.plot(x_as, shittoplot)
# sns.lineplot(x = "Episodes", y = "Rewards", data = df)
plt.show()

