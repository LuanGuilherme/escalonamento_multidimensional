from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import sklearn.datasets as dt
import seaborn as sns         
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('./dataset.csv')
X = np.array([df['VelocidadeMaxima'], df['HorarioH'], df['Idade'], df['Ano'], df['Mes']])
#X = preprocessing.normalize(X)
dist_euclid = euclidean_distances(X)

print("Distancia Euclidiana")
print(dist_euclid)


mds = MDS(random_state=0)
X_transform = mds.fit_transform(X)

print("Stress: ")
print(mds.stress_)

dist_manhattan = manhattan_distances(X)
print("Distancia manhattan")
print(dist_manhattan)

mds = MDS(dissimilarity='precomputed', random_state=0)
# Get the embeddings
X_transform_L1 = mds.fit_transform(dist_manhattan)


colors = ['r', 'g', 'b', 'c', 'm']
size = [64, 64, 64, 64, 64]
fig = plt.figure(2, (10,4))
ax = fig.add_subplot(121, projection='3d')
plt.scatter(X[:,0], X[:,1], zs=X[:,2], s=size, c=colors)
plt.title('Posições originais')
plt.savefig('Imagens/posicoesOriginais.png')
plt.close()

ax = fig.add_subplot(122)
pts = mds.fit_transform(dist_euclid)
ax = sns.scatterplot(x=pts[:, 0], y=pts[:, 1], hue=['VelocidadeMaxima', 'Horário', 'Idade','Ano', 'Mes'], palette=['r', 'g', 'b', 'c','m'])
plt.scatter(X_transform[:,0], X_transform[:,1], s=size, c=colors)
plt.title('Gráfico de duas dimensões')
fig.subplots_adjust(wspace=.4, hspace=0.5)
plt.savefig('Imagens/grafico2deuclides.png')
plt.close()


ax = fig.add_subplot(122)
pts = mds.fit_transform(dist_manhattan)
ax = sns.scatterplot(x=pts[:, 0], y=pts[:, 1], hue=['VelocidadeMaxima', 'Horário', 'Idade','Ano', 'Mes'], palette=['r', 'g', 'b', 'c','m'])
plt.scatter(X_transform_L1[:,0], X_transform_L1[:,1], s=size, c=colors)
plt.title('Gráfico de duas dimensões')
fig.subplots_adjust(wspace=.4, hspace=0.5)
plt.savefig('Imagens/grafico2dmanhattan.png')
plt.close()

stress = []
# Max value for n_components
max_range = 6
for dim in range(1, max_range):
    # Set up the MDS object
    mds = MDS(n_components=dim, dissimilarity='precomputed', random_state=0)
    # Apply MDS
    pts = mds.fit_transform(dist_euclid)
    # Retrieve the stress value
    stress.append(mds.stress_)
# Plot stress vs. n_components    
plt.plot(range(1, max_range), stress)
plt.xticks(range(1, max_range, 2))
plt.xlabel('Dimensões')
plt.ylabel('Estresse')
plt.savefig('Imagens/stress.png')
plt.close()