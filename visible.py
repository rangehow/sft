from cuml import UMAP
import torch

tensor_np=torch.rand((256,256000))
# 使用t-SNE降维到2D
tsne = UMAP(n_components=50)
tensor_2d = tsne.fit_transform(tensor_np)

# 绘制降维后的数据
plt.scatter(tensor_2d[:, 0], tensor_2d[:, 1])
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('t-SNE of High-Dimensional Tensor')
plt.show()
