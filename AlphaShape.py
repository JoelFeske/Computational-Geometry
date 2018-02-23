#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 17:01:58 2018

@author: Joel Feske
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay


class AlphaShape:
    
    '''
    
    https://en.wikipedia.org/wiki/Alpha_shape
    
    params: optional parameters in brackets
        delaunay: a Delaunay triangulation of a set of points using scipy.spatial.Delaunay
        [alpha]: determines which triangles in the Delaunay triangulation  are
                 included in the alpha shape. If none is supplied, one will be
                 calculated automatically at a reasonable scale. Ultimately this
                 parameter is arbitrary, and will depend on the application.
    
    '''
    
    def __init__(self, delaunay, alpha = None):
        if alpha is None:
            self.alpha = self.get_alpha(delaunay.points)
        else:
            self.alpha = alpha
        self.delaunay = delaunay
        self.alpha_complex = self.get_alpha_complex(self.alpha, delaunay)
        self.boundary_simplices = self.get_boundary_simplices(self.alpha, delaunay)
        self.boundary_points = self.get_boundary_points(self.boundary_simplices)
    
    # Returns the distance between the points in 'points' at the indices 'a' and 'b'
    def distance(self, a, b, points):
        x1 = points[a]
        x2 = points[b]
        d = np.sqrt((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2)
        return d
    
    def get_alpha(self, points):
        
        # Convert Cartesian coordinates to polar coordinates
        def to_polar(X):
            r = np.sqrt(X[0]**2 + X[1]**2)
            theta = np.arctan2(X[1], X[0])
            return r, theta
        
        r_max = float('-inf')
        r_min = float('inf')
        
        for point in points:
            r, _ = to_polar(point)
            if r > r_max:
                r_max = r
            if r < r_min:
                r_min = r
        
        r_diff = r_max - r_min
        alpha = 16/r_diff
        return alpha
        
    # 'delaunay' is some Delaunay triangulation of a set of points
    def get_alpha_complex(self, alpha, delaunay):
        # Find the circumradius of each triangle in the triangulation
        alpha_simplices = np.array([], dtype=int)
        for simplex in delaunay.simplices:
            # If the circumradius <= 1/alpha, include it in the alpha complex
            
            # vertices (indices of vertices)
            A, B, C = simplex[0], simplex[1], simplex[2]
            
            # side lengths
            a = self.distance(B, C, delaunay.points)
            b = self.distance(A, C, delaunay.points)
            c = self.distance(A, B, delaunay.points)
            
            # circumradius
            # http://mathworld.wolfram.com/Circumradius.html
            s = (a + b + c)/2
            R = (a*b*c)/(4*np.sqrt(s*(a+b-s)*(a+c-s)*(b+c-s)))
            if R <= 1/alpha:
                alpha_simplices = np.append(alpha_simplices, [simplex])
        
        # reshape to be m x 3
        return np.reshape(alpha_simplices, (int(len(alpha_simplices)/3), 3))
    
    # Filter out simplices with less than 3 neighbors (interior simplices)
    def get_boundary_simplices(self, alpha, delaunay):
        neighbors = delaunay.neighbors
        alpha_simplices = np.array([], dtype=int)
        alpha_neighbors = np.zeros(neighbors.shape, dtype=int)
        boundary_simplices = np.array([], dtype=int)
        # Find the circumradius of each triangle in the triangulation
        for simplex in delaunay.simplices:
            # If the circumradius <= 1/alpha, include it in the alpha complex
            
            # vertices (indices of vertices)
            A, B, C = simplex[0], simplex[1], simplex[2]
            
            # side lengths
            a = self.distance(B, C, delaunay.points)
            b = self.distance(A, C, delaunay.points)
            c = self.distance(A, B, delaunay.points)
            
            # circumradius
            # http://mathworld.wolfram.com/Circumradius.html
            s = (a + b + c)/2
            R = (a*b*c)/(4*np.sqrt(s*(a+b-s)*(a+c-s)*(b+c-s)))
            if R <= 1/alpha:
                alpha_simplices = np.append(alpha_simplices, [simplex])
            else:
                # this denotes a simplex that is not in the alpha complex
                alpha_simplices = np.append(alpha_simplices, [-1, -1, -1])
        # reshape to be m x 3
        alpha_simplices = np.reshape(alpha_simplices, (int(len(alpha_simplices)/3), 3))
        
        # fill the 'alpha_neighbors' array
        for i in range(len(neighbors)):
            for j in range(3):
                if neighbors[i][j] == -1:
                    alpha_neighbors[i][j] = -1
                elif np.sum(alpha_simplices[neighbors[i][j]]) == -3:
                    # this neighbor didn't make it into the alpha complex
                    alpha_neighbors[i][j] = -1
                else:
                    alpha_neighbors[i][j] = neighbors[i][j]
        
        # filter out placeholder entries of [-1, -1, -1]            
        alpha_simplices_filtered = np.array([simplex for simplex in alpha_simplices if np.sum(simplex) > -3])
        alpha_neighbors_filtered = np.array([alpha_neighbors[i] for i in range(len(alpha_simplices)) if np.sum(alpha_simplices[i]) > -3])
        
        # only include a simplex in the boundary if it has at least one neighbor
        # denoted as '-1' -- that is, if it has fewer than 3 neighbors    
        boundary_simplices = np.array([ alpha_simplices_filtered[i] for i in range(len(alpha_simplices_filtered)) if -1 in alpha_neighbors_filtered[i] ])
        
        return boundary_simplices
    
    # This is ROUGHLY correct, but will not always return the exact boundary points.
    # It will miss points that make their own corner of the shape
    def get_boundary_points(self, boundary_simplices):
        boundary_points = np.array([], dtype=int)
        unique, counts = np.unique(boundary_simplices, return_counts=True)
        for i in range(len(boundary_simplices)):
            for j in range(3):
                # get index of this point within 'unique'
                unique_index = int(np.where(unique == boundary_simplices[i][j])[0])
                # if 'counts[index]' > 1, add it to boundary_points. This means
                # that it is present in multiple boundary simplices.
                # Also, do not add a point if it is already present
                if counts[unique_index] == 2 and boundary_simplices[i][j] not in boundary_points:
                    boundary_points = np.concatenate((boundary_points, [boundary_simplices[i][j]]))
        return boundary_points
    
if __name__ == "__main__":
    
    # test data
    npts = 100#14300
    points_0 = np.random.random((npts,2))
    x_shift = np.array([np.array([1, 0]) for i in range(npts)])
    y_shift = np.array([np.array([0, 1]) for i in range(npts)])
    points_1 = np.random.random((npts,2)) + y_shift
    points_2 = np.random.random((npts,2)) + 2*y_shift
    points_3 = np.random.random((npts,2)) + x_shift + y_shift
    points_4 = np.random.random((npts,2)) + 2*x_shift
    points_5 = np.random.random((npts,2)) + 2*x_shift + y_shift
    points_6 = np.random.random((npts,2)) + 2*x_shift + 2*y_shift
    points = np.concatenate((points_0, points_1, points_2, points_3, points_4, points_5, points_6))
    #points = 100*points
    
    points2 = points + 3*np.array([np.array([1, 0]) for i in range(len(points))]) + 3*np.array([np.array([0, 1]) for i in range(len(points))])
    
    # Delaunay
    tri = Delaunay(points)
    
    a_shape = AlphaShape(tri)
    alpha_simplices = a_shape.alpha_complex

    boundary_simplices = a_shape.boundary_simplices
    
    boundary_points = a_shape.boundary_points
    boundary_coords = []
    for i in range(len(boundary_points)):
        boundary_coords.append(points[boundary_points[i]])
    boundary_coords = np.array(boundary_coords)
    
    colors = np.array([points[simplex[0]][1] for simplex in alpha_simplices])
    colors = np.array([0 for simplex in alpha_simplices])
    #colors = np.random.random(alpha_simplices.shape[0])
        
    colors2 = np.array([.5 for simplex in alpha_simplices])
    
    # fill test
    fill = np.array([[0,0], [3,0], [1.5,3]])#, [0,0]])
    fill = np.array([[0,0], [1,0], [1,1], [2,1], [2,0], [3,0], [3,3], [2,3], [2,2], [1,2], [1,3], [0,3]])
    
    # Plot

    # plot alpha shape
    plt.triplot(points[:,0], points[:,1], alpha_simplices, color = 'm')

    # plot boundary simplices
    plt.triplot(points[:,0], points[:,1], boundary_simplices, color = 'c')
    
    # plot boundary points
    plt.scatter(boundary_coords[:,0], boundary_coords[:,1], color='k')
        
    plt.show()