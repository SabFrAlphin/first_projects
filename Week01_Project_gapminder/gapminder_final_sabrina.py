import pandas as pd
import pylab as plt
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
sns.set()

fert = pd.read_csv('gapminder_total_fertility.csv', index_col=0)
life = pd.read_excel('gapminder_lifeexpectancy.xlsx', index_col=0)
pop = pd.read_excel('gapminder_population.xlsx', index_col=0)

ncol = [int(x) for x in fert.columns]
fert.set_axis(axis=1, labels=ncol, inplace=True)
fert.columns

sfert = fert.stack()
slife = life.stack()
spop = pop.stack()

d = {'fertility': sfert, 'lifeexp': slife, 'population': spop}
df = pd.DataFrame(data=d)

gm = df.stack().unstack(1).unstack(1)
for i in range(1960, 2016):
    gm1 = gm[i]
    cmap = plt.get_cmap('tab20b', lut=len(gm1)).colors
    gm1.plot.scatter('fertility', 'lifeexp', s=df['population'] / 100000, c=cmap)
    plt.axis((1, 10, 10, 90))
    plt.title(f'{i}')
    plt.savefig(f'scatter_images/lifeexp_{i}.png', bbox_inches='tight')
    plt.clf()

images = []
for i in range(1960, 2016):
    filename = 'scatter_images/lifeexp_{}.png'.format(i)
    images.append(imageio.imread(filename))

imageio.mimsave('output_gapminder.gif', images, fps=10)
