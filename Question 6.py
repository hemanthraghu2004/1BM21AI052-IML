import numpy as np
from sklearn.mixture import GaussianMixture

np.random.seed(42)
data = np.concatenate([np.random.normal(-5, 1, 300), np.random.normal(5, 1, 300)]).reshape(-1, 1)

gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(data.reshape(-1, 1))

predicted_labels = gmm.predict(data.reshape(-1, 1))
means = gmm.means_

print("Predicted Labels:", predicted_labels)
print("Means:", means)
