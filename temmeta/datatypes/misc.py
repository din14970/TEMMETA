
class PointCloud(object):
    """
    Represents an n-dimensional scalar point cloud

    Points have coordinates in n-dimensional space. They can have multiple
    associated scalar values.
    """
    def __init__(self, coordinates, scalars, metadata):
        self.__coordinates = coordinates
        self.__scalars = scalars

    @property
    def coordinates(self):
        return self.coordinates

    @property
    def scalars(self):
        return self.scalars

    def calculate_nn(self):
        pass


class NearestNeighborNetwork(object):

    def __init__(self):
        pass

    @property
    def X(self):
        """X coordinates of nearest neighbors for each point"""
        return self.data[0]

    @property
    def Y(self):
        """Y coordinates of nearest neighbors for each point"""
        return self.data[1]

    @property
    def angle(self):
        """angle of nearest neighbors for each point"""

    @property
    def distance(self):
        """distance of nearest neighbors for each point"""
