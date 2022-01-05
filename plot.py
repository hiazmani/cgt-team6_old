import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./trappedBox12k_length.csv")

window = 10

episodes = df['Episodes'].to_numpy()
rewards = df['AmountOfSteps'].to_numpy()
print(episodes)
# rewards = df.y
x_as = []
shittoplot = []
n_episodes = len(episodes)
div = n_episodes // window
for step in range(div):
    curr_av = 0
    for x in range(window):
        curr_av += rewards[(step*window)+x]
    x_as.append(step * window)
    shittoplot.append(curr_av / window)     
    
plt.plot(x_as, shittoplot)
# sns.lineplot(x = "Episodes", y = "Rewards", data = df)
plt.show()

