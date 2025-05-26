from point import makePointList, Point


class Cluster:
    """A class representing a cluster of points.

    Attributes:
      center: A Point object representing the exact center of the cluster.
      points: A set of Point objects that constitute our cluster.
    """

    def __init__(self, center=Point([0, 0])):
        """Inits a Cluster with a specific center (defaults to [0,0])."""
        self.center = center
        self.points = set()

    @property
    def coords(self):
        return self.center.coords

    @property
    def dim(self):
        return self.center.dim

    def addPoint(self, p):
        self.points.add(p)

    def removePoint(self, p):
        self.points.remove(p)

    @property
    def avgDistance(self):
        """Calculates the average distance of points in the cluster to the center.

        Returns:
          A float representing the average distance from all points in self.points
          to self.center.
        """
        # fill in
        
        # take the number of points
        point_count = len(self.points) # note that this is extremely redundant

        # make a running distance count
        total_distance = 0

        # iterate through each point and sum the distance to the center
        for p in self.points:

            # use the distFrom method to find the ith distance at point p
            p_dist = p.distFrom(self.center)

            # add the ith distance to the running total
            total_distance += p_dist

        # compute the mean distance (divide by Point count)
        mean_distance = total_distance / point_count
        
        # return the mean distance
        return mean_distance

    def updateCenter(self):
        """Updates self.center to be the average of all points in the cluster.

        If no points are in the cluster, then self.center should be unchanged.

        Returns:
          The coords of self.center.
        """
        # fill in
        # Hint: make sure self.center is a Point object after this function runs.
        # check empty 
        if len(self.points) == 0:
            self.center = Point(self.coords)
            return self.center.coords
        
        # initialize coordiantes of center
        c_coords = []

        # for each d in the dimension, average the values across p in points
        p_, d_ = len(self.points), self.dim

        for d in range(d_):

            # have a running sum in the coordiante
            d_sum = 0.0

            # iterate through each Point and add to the sum
            for p in self.points:
                
                # capture the coord in the d position
                d_coord = float(p.coords[d])

                # add to the dth sum
                d_sum += d_coord

            # take the dth average
            d_average = d_sum / p_

            # append this as a new coordiante
            c_coords.append(d_average)

        # move the center
        self.center = Point(c_coords)

        # return the center coords
        return self.center.coords

    def printAllPoints(self):
        print(str(self))
        for p in self.points:
            print("   {}".format(p))

    def __str__(self):
        return "Cluster: {} points and center = {}".format(
            len(self.points), self.center
        )

    def __repr__(self):
        return self.__str__()


def createClusters(data):
    """Creates clusters with centers from a k-by-d numpy array.

    Args:
      data: A k-by-d numpy array representing k d-dimensional points.

    Returns:
      A list of Clusters with each cluster centered at a d-dimensional
      point from each row of data.
    """
    centers = makePointList(data)
    return [Cluster(c) for c in centers]


if __name__ == "__main__":

    p1 = Point([2.5, 4.0])
    p2 = Point([3.0, 5.0])
    p3 = Point([1.0, 3.0])
    c = Cluster(Point([2.0, 2.0]))
    print(c)

    c.addPoint(p1)
    c.addPoint(p2)
    c.addPoint(p3)
    print(c)
    print(c.avgDistance)
    c.updateCenter()
    print(c)
    print(c.avgDistance)
    assert isinstance(
        c.center, Point
    ), "After updateCenter, the center must remain a Point object."
