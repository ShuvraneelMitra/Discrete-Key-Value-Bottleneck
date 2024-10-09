import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

NUM_CLASSES = 8
SAMPLES_PER_CLASS = 100 
FEAT_DIMENSION = 2 
RAD = 10  

data = []
labels = []

angles = np.linspace(0, 2 * np.pi, NUM_CLASSES, endpoint=False)  

for i, angle in enumerate(angles):
    
    mean = np.array([RAD * np.cos(angle), RAD * np.sin(angle)])
    mean += np.random.normal(loc=0.0, scale=1.0, size=FEAT_DIMENSION)
    
    # np.diag() is chosen on the assumption that the two features are statistically independent
    cov = np.diag(np.random.uniform(low=0.5, high=1.5, size=FEAT_DIMENSION))

    class_data = np.random.multivariate_normal(mean, cov, SAMPLES_PER_CLASS)

    data.append(class_data)
    labels.append([i] * SAMPLES_PER_CLASS)

data = np.vstack(data)
labels = np.hstack(labels)

df = pd.DataFrame(data)
df["Labels"] = labels
df.columns = ["Feature_1", "Feature_2", "Labels"]

df.to_csv("Toy Synthetic Dataset.csv")

# Visualization: uncomment the below lines to view the generated dataset
# plt.figure(figsize=(8, 8))
# for i in range(NUM_CLASSES):
#     plt.scatter(df[df["Labels"] == i]["Feature_1"], df[df["Labels"] == i]["Feature_2"], 
#                 label=f'Class {i}')
# plt.legend()
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('2D Features with 8 classes')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()
