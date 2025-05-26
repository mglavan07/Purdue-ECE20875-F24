import numpy as np
from cluster import createClusters
from point import makePointList


def kmeans(point_data, cluster_data):
    """Performs k-means clustering on points.

    Args:
      point_data: a p-by-d numpy array used for creating a list of Points.
      cluster_data: A k-by-d numpy array used for creating a list of Clusters.

    Returns:
      A list of clusters (with update centers) after peforming k-means
      clustering on the points initialized from point_data
    """
    # Fill in

    # 1. Make list of points using makePointList and point_data
    points = makePointList(point_data)

    # 2. Make list of clusters using createClusters and cluster_data
    clusters = createClusters(cluster_data)

    # 3. For as long as points keep moving between clusters:

    # allow the while loop to enter
    repeat = True

    # improvised do-while loop
    while repeat: # booleans contains at least a 1, meaning moved was True at some point

    #   A. Move every point to its closest cluster (use Point.closest and
    #     Point.moveToCluster)
    #     Hint: keep track here whether any point changed clusters by
    #           seeing if any moveToCluster call returns "True"

      # make a list of booleans for the "moved" return
      repeat = False
    
      for point in points:

        # find the closest cluster
        nearest = point.closest(clusters)

        # move to this center
        moved = point.moveToCluster(nearest)

        # determine if the loop should exit
        if moved:
          repeat = True

    #   B. Update the centers for each cluster (use Cluster.updateCenter)
      for cluster in clusters:
         cluster.updateCenter()

    # 4. Return the list of clusters, with the centers in their final positions
    return clusters


if __name__ == "__main__":
    data = np.array(
        [
            [12.1, -7.1], [0.5, 2.8], [1.2, 5.3], [10.3, -4.8], [-1.1, 3.9],
            [8.9, -3.6], [11.5, -6.2], [7.4, -2.5], [10.8, -5.5], [9.4, -4.3]
        ],
        dtype=float,
    )
    centers = np.array([[0, 0], [1, 1]], dtype=float)

    clusters = kmeans(data, centers)
    for c in clusters:
        c.printAllPoints()
