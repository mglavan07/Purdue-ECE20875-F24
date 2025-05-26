import math
import numpy as np


class Point:
    """An n-dimensional Point.

    Attributes:
      coords: A list of length n specifying each coordinate of the Point.
      currCluster: A reference to the Cluster object the Point is in.
    """

    def __init__(self, coords):
        """Initializes a Point with a list of coordinates."""

        self.coords = coords
        self.currCluster = None

    @property
    def dim(self):
        return len(self.coords)

    def distFrom(self, other):
        """Calculates distance between two Points.

        Args:
          other: The Point we are calculating the distance from.

        Returns:
          A float representing the Euclidean distance between this point and other.
        """
        # Error checking, keep this here. - ensures Point and other have same dimension
        if self.dim != other.dim:
            raise ValueError(
                "dimension mismatch: self has dim {} and other has dim {}".format(
                    self.dim, other.dim
                )
            )

        # Hint: Refer to the formula given in README.md for the Euclidean distance

        # fill in

        # access the dimension of the point
        dimension = self.dim
        euclidian = 0

        # iterate through each coordinate in the dimension of point and other
        for i in range(dimension):
            
            # find the difference in the cooresponsing coordinates
            difference = self.coords[i] - other.coords[i]

            # add the squared difference to euclidian
            euclidian += difference ** 2

        # return the square root of the total squared difference
        return math.sqrt(euclidian)

    def moveToCluster(self, dest):
        """Reassigns this Point to a new Cluster.

        Args:
          dest: A Cluster object the Point will move to.

        Returns:
          True if dest is different from the current cluster, False otherwise.
        """
        if self.currCluster is dest:
            return False
        else:
            if self.currCluster:
                self.currCluster.removePoint(self)
            dest.addPoint(self)
            self.currCluster = dest
            return True

    def closest(self, objects):
        """Return the object that is closest to this point.

        Args:
          objects: A list of objects.

        Returns:
          The object in objects that is closest to this point. This
          object can be a Cluster or a Point.
        """
        minDist = self.distFrom(objects[0])
        minPt = objects[0]
        for p in objects:
            if self.distFrom(p) < minDist:
                minDist = self.distFrom(p)
                minPt = p
        return minPt

    def __getitem__(self, i):
        """p[i] will get the ith coordinate of the Point p."""
        return self.coords[i]

    def __str__(self):
        return str(self.coords)

    def __repr__(self):
        return f"Point({self.__str__()})"


def makePointList(data):
    """Creates a list of points from initialization data.
    #This function is outside Point Class.
    Args:
      data: A p-by-d numpy array.

    Returns:
      A list of length p containing d-dimensional Point objects, each Point's
      coordinates correspond to one row of data.
    """
    # fill in

    # initialize an empty list
    pd_list = []
    
    # define the number of points (p) and dimension of each (d)
    p, d = data.shape
    
    # intsantize a Point for each p, each point has dimension d
    for i in range(p):
        
        # slice the necessary data
        coordinates = list(data[i, :])

        # make a Point instance
        point_i = Point(coordinates)

        # add the point to pd_list
        pd_list.append(point_i)

    # return the list of p Points
    return pd_list


if __name__ == "__main__":
    data = np.array(
        [[0.5, 2.5], [0.3, 4.5], [-0.5, 3], [0, 1.2], [10, -5], [11, -4.5], [8, -3]]
    )

    points = makePointList(data)
    print(points)

    print(points[0].distFrom(points[1]))
