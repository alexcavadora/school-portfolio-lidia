import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.loadtxt("output/lata.csv", delimiter=',', skiprows=1)
data1 = data[0 : 79]
data2 = data[80 : 159]
data3 = data[160 : 239]
data4 = data[240 : 319]
data5 = data[320 : 399]
data6 = data[400 : 479]
data7 = data[480 : 549]
data8 = data[550 :]
print(data8)

# Extraer características
# AABB_Width,AABB_Height,AABB_Depth
aabbWidth = data[:, 4]
aabbHeight = data[:, 5]
aabbDepth = data[:, 6]
#OBB_Width,OBB_Height,OBB_Depth
obbWidth = data[:, 7]
obbHeight = data[:, 8]
obbDepth = data[:, 9]

major_axis_horizontal=data[:, 10]
major_axis_vertical=data[:, 11]

fig, axs = plt.subplots(2, 2, figsize=(8,8))

axs[0][0].scatter(aabbWidth, obbWidth, alpha=0.5, c='blue', s=1)
axs[0][0].set_xlabel('Axis Aligned bounding box width')
axs[0][0].set_ylabel('Object Aligned bounding box width ')
axs[0][0].set_title('AABB width vs OBB width')

#%%
# Gráfico 1: Característica 1 vs Característica 2
axs[0][1].scatter(aabbHeight, obbHeight, alpha=0.5, c='red', s=1)
axs[0][1].set_xlabel('Axis Aligned bounding box height')
axs[0][1].set_ylabel('Object Aligned bounding box height ')
axs[0][1].set_title('AABB height vs OBB height')


axs[1][0].scatter(aabbDepth, obbDepth, alpha=0.5, c='green', s=1)
axs[1][0].set_xlabel('Axis Aligned bounding box depth')
axs[1][0].set_ylabel('Object Aligned bounding box depth ')
axs[1][0].set_title('AABB depth vs OBB depth')


axs[1][1].scatter(major_axis_horizontal, major_axis_vertical, alpha=0.5, c='purple', s=1)
axs[1][1].set_xlabel('Major axis horizontal')
axs[1][1].set_ylabel('Major axis vertical')
axs[1][1].set_title('Major axis horizontal vs Major axis vertical')

aabbWidth = data2[:, 4]
aabbHeight = data2[:, 5]
aabbDepth = data2[:, 6]

obbWidth = data2[:, 7]
obbHeight = data2[:, 8]
obbDepth = data2[:, 9]
major_axis_horizontal=data2[:, 10]
major_axis_vertical=data2[:, 11]

axs[0][0].scatter(aabbWidth, obbWidth, alpha=0.5, c='red', s=1)

axs[0][1].scatter(aabbHeight, obbHeight, alpha=0.5, c='blue', s=1)

axs[1][0].scatter(aabbDepth, obbDepth, alpha=0.5, c='purple', s=1)

axs[1][1].scatter(major_axis_horizontal, major_axis_vertical, alpha=0.5, c='green', s=1)

plt.tight_layout()

plt.show()
