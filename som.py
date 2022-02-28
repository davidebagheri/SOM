import numpy as np


class SOM:
    def __init__(self, vector_dim, n_vectors, low, high):
        self.n_vectors = n_vectors
        self.centers = np.random.uniform(low, high, (n_vectors, vector_dim))

    def getClosestCenterIndex(self, sample):
        diff = np.linalg.norm(self.centers - sample, axis=1)
        center_idx = np.argmin(diff)
        return center_idx

    def getClosestCenter(self, sample):
        center_idx = self.getClosestCenterIndex(sample)
        return self.centers[center_idx]

    def getCentersInNeighbourhood(self, center_id, radius):
        neigh_centers_ids = []
        neig_dist = []
        ref_center = self.centers[center_id]

        diff = np.linalg.norm(self.centers - ref_center, axis=1)

        for i in range(self.n_vectors):
            if diff[i] <= radius:
                neigh_centers_ids.append(i)
                neig_dist.append(diff[i])
        return neigh_centers_ids, neig_dist


    def train_step(self, sample, radius, learning_rate):
        bmu_id = self.getClosestCenterIndex(sample)

        neigh_centers_ids, neig_dist = self.getCentersInNeighbourhood(bmu_id, radius)

        for id, dist in zip(neigh_centers_ids, neig_dist):

            if (id == bmu_id):
                distance_rate = np.exp(- np.power(dist, 2) / (2 * np.power(radius, 2)))
                self.centers[id] += learning_rate * distance_rate * (sample - self.centers[id])
            else:
                dist_to_bmu = np.linalg.norm(self.centers[bmu_id] - self.centers[id])
                distance_rate = np.exp(- np.power(dist_to_bmu, 2) / (2 * np.power(radius, 2)))
                self.centers[id] += learning_rate * distance_rate * (self.centers[id] - self.centers[bmu_id])

