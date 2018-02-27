# Computational-Geometry
Some classes I've written to solve a few computational geometry problems I've been working on -- see a full explanation here: http://www.joelfeske.com/computational-geometry.html

KMeans:
This is an extension of the standard KMeans algorithm. It runs a standard KMeans clustering, but there is the option to equalize the sizes of the clusters afterwards.

Next steps: Implement kmeans++ (https://en.wikipedia.org/wiki/K-means%2B%2B) for initializing the means.

Note: Currently there is not a way to guarantee that the equalized clusters
      are contiguous. See the Delaunay_Groups.AlphaCluster class which
      partially addresses this problem.

AlphaShape:
Returns a triangulation of the alpha shape of a set of points based on a Delaunay triangulation of that point set.

https://en.wikipedia.org/wiki/Delaunay_triangulation
https://en.wikipedia.org/wiki/Alpha_shape

DelaunayGroups:
Contains the classes:
  AlphaGroups: like AlphaShape -- see comments for details.
  AlphaCluster: A contiguous cluster of triangles within the AlphaGroup (alpha shape). This class is intended to address the problem noted in the KMeans class
