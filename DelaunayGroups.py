#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 04:43:07 2018

@author: Joel Feske
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

from KMeans import KMeans
#from AlphaShape import AlphaShape

class AlphaGroups:
    
    '''
    
    This is essentially the same as the AlphaShape class, except that it does
    not filter out the Delaunay triangles that are not included in the alpha shape.
    This is so that the indices of all the Delaunay triangles are maintained
    and can be referenced later. 
    
    https://en.wikipedia.org/wiki/Alpha_shape
    
    params: optional parameters in brackets
        delaunay: a Delaunay triangulation of a set of points using scipy.spatial.Delaunay
        [alpha]: determines which triangles in the Delaunay triangulation  are
                 included in the alpha shape. If none is supplied, one will be
                 calculated automatically at a reasonable scale. Ultimately this
                 parameter is arbitrary, and will depend on the application.
    
    '''
    
    def __init__(self, delaunay, alpha = None):
#        self.alpha = self.get_alpha(self.points)
        self.points = delaunay.points
        if alpha is None:
            self.alpha = self.get_alpha(self.points)
        else:
            self.alpha = alpha
        self.delaunay = delaunay
        self.alpha_complex, self.alpha_neighbors = self.get_alpha_complex_and_neighbors(self.alpha, delaunay)
    
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
    
    # Filter out simplices with less than 3 neighbors (interior simplices)
    def get_alpha_complex_and_neighbors(self, alpha, delaunay):
        neighbors = delaunay.neighbors
        alpha_simplices = np.array([], dtype=int)
        alpha_neighbors = np.zeros(neighbors.shape, dtype=int)
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
#        alpha_simplices_filtered = np.array([simplex for simplex in alpha_simplices if np.sum(simplex) > -3])
#        alpha_neighbors_filtered = np.array([alpha_neighbors[i] for i in range(len(alpha_simplices)) if np.sum(alpha_simplices[i]) > -3])
        
        return alpha_simplices, alpha_neighbors
#        return alpha_simplices_filtered, alpha_neighbors_filtered

class AlphaCluster:
    
    '''
    Throughout this class, simplices are referred to by their index number. This is
    their index number in the original Delaunay triangulation. Their index number
    within the array self.simplex_indices in this class must be calculated using
    numpy.where(self.simplex_index == desired_index)[0][0], where we reference the [0][0]
    element of the output array of numpy.where().
    
    params:
        alpha_shape: an object of type AlphaGroups (NOT AlphaShape)
        [simplex_indices]: the first simplex to add to the cluster. This is optional,
                           as you can create an empty alpha cluster with no simplices
    
    '''
    
    # simplex_indices are the indices of the simplices included in this cluster.
    # Indices of vertices can be found in alpha_neighbors
    # Coordinates of vertices are found in the original data array
    def __init__(self, alpha_shape, simplex_indices = np.array([], dtype=int)):
        
        self.alpha_complex = alpha_shape.alpha_complex
        self.alpha_neighbors = alpha_shape.alpha_neighbors
        self.simplex_indices = simplex_indices
        self.neighbors = self.initialize_neighbors()#np.array([], dtype=int)
        self.frontier_indices = self.initialize_frontier()#self.update_frontier() #simplices adjacent to, but not in the cluster
        
    def initialize_neighbors(self):
        n_simplices = len(self.simplex_indices)
        if n_simplices == 0:
            return np.array([], dtype=int)
        if n_simplices == 1:
            return np.array([-1, -1, -1], dtype=int)
        else:
            print("Cannot initialize AlphaCluster with more than one simplex")
            return None
        
    def initialize_frontier(self):
        n_simplices = len(self.simplex_indices)
        if n_simplices == 0:
            return np.array([], dtype=int)
        if n_simplices == 1:
            return np.array([ alpha_neighbor for alpha_neighbor in self.alpha_neighbors[self.simplex_indices[0]] if alpha_neighbor != -1])
        else:
            print("Cannot initialize AlphaCluster with more than one simplex")
            return None
    
    def is_in_boundary(self, simplex_index):
        # get index within the self.simplex_indices array
        simplex_index_index = np.where(self.simplex_indices == simplex_index)[0][0]
        neighbors = self.neighbors[simplex_index_index]
        return -1 in neighbors
    
    def get_boundary(self):
        boundary_simplices = np.array([], dtype=int)
        for simplex_index in self.simplex_indices:
            if self.is_in_boundary(simplex_index):
                boundary_simplices = np.append(boundary_simplices, simplex_index)
        return boundary_simplices
    
    def add_simplex(self, added_simplex_index):
        if self.frontier_indices.size > 0 and added_simplex_index not in self.frontier_indices:
            print("Cannot add simplex that is not in frontier")
            return
        else:
            self.simplex_indices = np.append(self.simplex_indices, added_simplex_index)
            # FIX THIS
            # the cleanup functions are much slower than the update functions
            self.cleanup_neighbors()
#            self.update_neighbors_add(added_simplex_index)
            self.update_frontier_add(added_simplex_index)
            self.cleanup_neighbors()
#            self.cleanup_frontier()
    
    def remove_simplex(self, removed_simplex_index):
        if self.is_in_boundary(removed_simplex_index):
            # remove
            removed_simplex_index_index = np.where(self.simplex_indices == removed_simplex_index)[0][0]
            self.simplex_indices = np.delete(self.simplex_indices, removed_simplex_index_index)
            # update neighbors
            self.cleanup_neighbors()
            # update frontier
            self.cleanup_frontier()
        else:
            print("Cannot remove simplex not in boundary.")
    
    # This is for fixing self.neighbors after a simplex is removed from the cluster
    # Seems unnecessarily expenive 
    def cleanup_neighbors_old(self):
        clean_neighbors = np.array([], dtype=int)
        for simplex_index_index, simplex_index in enumerate(self.simplex_indices):
            neighbors = np.array([ neighbor if neighbor in self.simplex_indices else -1 for neighbor in self.alpha_neighbors[simplex_index] ])
            clean_neighbors = np.append(clean_neighbors, neighbors)
        self.neighbors = clean_neighbors
        
    def cleanup_neighbors(self):
        clean_neighbors = np.array([], dtype=int)
        for simplex_index_index, simplex_index in enumerate(self.simplex_indices):
            neighbors = np.array([ neighbor if neighbor in self.simplex_indices else -1 for neighbor in self.alpha_neighbors[simplex_index] ])
            if len(clean_neighbors) == 0:
                clean_neighbors = np.append(clean_neighbors, neighbors)
            else:
                clean_neighbors = np.vstack((clean_neighbors, neighbors))
        self.neighbors = clean_neighbors
                    
    
    # This is for fixing the frontier after a simplex is removed from the cluster
    # Seems unnecessarily expenive 
    def cleanup_frontier(self):
        clean_frontier = np.array([], dtype=int)
        for simplex_index in self.simplex_indices:
            if self.is_in_boundary(simplex_index):
                neighbors = self.alpha_neighbors[simplex_index]
                for neighbor in neighbors:
                    if neighbor != -1 and neighbor not in self.simplex_indices:
                        clean_frontier = np.append(clean_frontier, neighbor)
        self.frontier_indices = clean_frontier
                
    def update_neighbors_add(self, added_simplex_index=None):
        # find alpha_neighbors of the added simplex
        possible_neighbors = self.alpha_neighbors[added_simplex_index]
        # filter out the ones not in the cluster
        added_simplex_neighbors = np.array([ neighbor if neighbor in self.simplex_indices else -1 for neighbor in possible_neighbors ])
        # add the neighbors for the added simplex
        if len(self.neighbors) == 0:
            self.neighbors = np.append(self.neighbors, [added_simplex_neighbors])
        else:
            self.neighbors = np.vstack((self.neighbors, added_simplex_neighbors))
        # update the neighbors for each alpha_neighbor that is in the cluster
        for added_neighbor in added_simplex_neighbors:
            # update the neighbors for this added_neighbor
            if added_neighbor != -1:
                # find index of the added_neighbor within self.simplex_indices
                added_neighbor_index = np.where(self.simplex_indices == added_neighbor)[0][0]
                print("added_neighbor_index = {}".format(added_neighbor_index))
                print("np.where(self.neighbors[added_neighbor_index] == -1) = {}".format(np.where(self.neighbors[added_neighbor_index] == -1)))
                self.neighbors[added_neighbor_index][np.where(self.neighbors[added_neighbor_index] == -1)[0][0]] = added_simplex_index
        
    def update_frontier_add(self, added_simplex_index=None):
        if self.frontier_indices.size > 0:
            # so... this line is inscrutable, but what it does is remove the added
            # simplex index from the frontier, since it's now part of the cluster
            self.frontier_indices = np.delete(self.frontier_indices, np.where(self.frontier_indices == added_simplex_index)[0][0])
        for alpha_neighbor in self.alpha_neighbors[added_simplex_index]:
            if alpha_neighbor not in self.simplex_indices \
            and alpha_neighbor not in self.frontier_indices \
            and alpha_neighbor != -1:
                self.frontier_indices = np.append(self.frontier_indices, alpha_neighbor)

# clustering functions

def initialize_cluster_seeds(k, alpha_shape):
    alpha_complex = alpha_shape.alpha_complex
    alpha_complex_filtered = np.array([ simplex_index for (simplex_index, simplex) in enumerate(alpha_complex) if np.sum(simplex) > -3])
    simplex_indices = np.array([ alpha_complex_filtered[int(n)] for n in np.floor(np.linspace(0, len(alpha_complex_filtered)-1, k)) ])
    clusters = [ AlphaCluster(alpha_shape, simplex_indices=np.array([simplex_indices[i]])) for i in range(len(simplex_indices)) ]
    return alpha_complex, alpha_complex_filtered, simplex_indices, clusters

def is_claimed(simplex_index, clusters):
    for cluster in clusters:
        for index in cluster.simplex_indices:
            if index == simplex_index:
                return True, cluster
    return False, None

# this could call is_claimed(), but then it would check the cluster that the
# frontier_index belonged to, which would be a waste
def first_unclaimed(cluster, clusters):
    for frontier_index in cluster.frontier_indices:
        claimed = False
        for cluster_check in clusters:
            if cluster != cluster_check and frontier_index in cluster_check.simplex_indices:
                claimed = True
                break
        if not claimed:
            return frontier_index
    return None

def grow_clusters(k, alpha_shape):
    alpha_complex = alpha_shape.alpha_complex
    cluster_size = len(alpha_complex)//k
    _,_,_,clusters = initialize_cluster_seeds(k, alpha_shape)
    
    smallest_cluster_size = 1
    count = 0
    while smallest_cluster_size < cluster_size:
        print(count)
        # add a simplex to each cluster
        for cluster in clusters:
            # check for the first unclaimed simplex in cluster.frontier_indices
            simplex_to_add = first_unclaimed(cluster, clusters)
            # if one exists, add it
            if simplex_to_add is not None:
                cluster.add_simplex(simplex_to_add)
            else:
                # else if non are unclaimed, add the first frontier index
                simplex_to_add = cluster.frontier_indices[0]
                _,claimed_by = is_claimed(simplex_to_add, clusters)
                # remove the simplex from whichever cluster claims it
                claimed_by.remove_simplex(simplex_to_add)
                # add it to this one
                cluster.add_simplex(simplex_to_add)
            
        
        # get new cluster sizes
        smallest_cluster_size = np.min([ len(cluster.simplex_indices) for cluster in clusters ])
        
        if count == 160:
            print("MAX COUNT REACHED")
            break
        count += 1
        
    return clusters

def distance(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def point_closest_to_mean(mean, points):
    min_dist = float('inf')
    closest_point = None
    for point in points:
        dist = distance(mean, point)
        if dist < min_dist:
            min_dist = dist
            closest_point = point
    return closest_point
    

def initialize_cluster_seeds_from_means(k, alpha_shape, means):
    points = np.array([ point_closest_to_mean(mean, alpha_shape.points) for mean in means ])
    simplex_indices = np.array([ alpha_shape.delaunay.find_simplex(point) for point in points ])
    clusters = [ AlphaCluster(alpha_shape, simplex_indices=np.array([simplex_indices[i]])) for i in range(len(simplex_indices)) ]
    return points, simplex_indices, clusters
    
def grow_clusters_from_means_old(k, alpha_shape, means):
    alpha_complex = alpha_shape.alpha_complex
    cluster_size = len(alpha_complex)//k
    _,_,clusters = initialize_cluster_seeds_from_means(k, alpha_shape, means)
    
    smallest_cluster_size = 1
    count = 0
    while smallest_cluster_size < cluster_size:
        print("===")
        print(count)
        print(cluster_size)
        print(smallest_cluster_size)
        # add a simplex to each cluster
        for cluster in clusters:
            # check for the first unclaimed simplex in cluster.frontier_indices
            simplex_to_add = first_unclaimed(cluster, clusters)
            # if one exists, add it
            if simplex_to_add is not None:
                cluster.add_simplex(simplex_to_add)
            else:
                # else if non are unclaimed, add the first frontier index
                simplex_to_add = cluster.frontier_indices[0]
                _,claimed_by = is_claimed(simplex_to_add, clusters)
                # remove the simplex from whichever cluster claims it
                claimed_by.remove_simplex(simplex_to_add)
                # add it to this one
                cluster.add_simplex(simplex_to_add)
            
        
        # get new cluster sizes
        smallest_cluster_size = np.min([ len(cluster.simplex_indices) for cluster in clusters ])
        
        if count == 300:
            print("MAX COUNT REACHED")
            break
        count += 1
        
    return clusters

def find_adjacent_clusters(cluster, clusters):
    pass

def grow_clusters_from_means(k, alpha_shape, means):
    alpha_complex = alpha_shape.alpha_complex
    cluster_size_target = len(alpha_complex)//k - 1
    _,_,clusters = initialize_cluster_seeds_from_means(k, alpha_shape, means)
    
    smallest_cluster_size = 1
    count = 0
    stop_possible = False
    percent_difference = float('inf')
    while percent_difference > 0.05:#smallest_cluster_size < cluster_size_target:
#        print("===")
#        print(count)
#        print(cluster_size_target)
#        print(smallest_cluster_size)
#        print([ len(cluster.simplex_indices) for cluster in clusters ])
        # add a simplex to each cluster
        for index, cluster in enumerate(clusters):
            cluster_size = len(cluster.simplex_indices)
            if cluster_size < cluster_size_target + 1:
#                print("adding simplex to cluster {}.".format(index))
                # check for the first unclaimed simplex in cluster.frontier_indices
                simplex_to_add = first_unclaimed(cluster, clusters)
                # if one exists, add it
                if simplex_to_add is not None:
                    cluster.add_simplex(simplex_to_add)
                else:
                    # add RANDOM frontier index
                    simplex_to_add = np.random.choice(cluster.frontier_indices, 1)[0]
                    _,claimed_by = is_claimed(simplex_to_add, clusters)
                    # remove the simplex from whichever cluster claims it
                    if claimed_by.is_in_boundary(simplex_to_add):
                        claimed_by.remove_simplex(simplex_to_add)
                        # add it to this one
                        cluster.add_simplex(simplex_to_add)
            elif cluster_size > cluster_size_target:
                # remove a random simplex
                # preferrably one that's in the frontier of an undersized cluster
                
                # just doing a random one for now
                pass
                
                
            
        
        # get new cluster sizes
        cluster_sizes = [ len(cluster.simplex_indices) for cluster in clusters ]
        smallest_cluster_size = np.min(cluster_sizes)
        avg_cluster_size = np.mean(cluster_sizes)
        if avg_cluster_size >= cluster_size_target - 5 \
        and avg_cluster_size <= cluster_size_target + 5:
            stop_possible = True
            percent_difference = (np.max(cluster_sizes) - np.min(cluster_sizes))/np.min(cluster_sizes)
            
        
        if count%1000 == 0:
            print("===")
            print("count = {}".format(count))
            print(cluster_sizes)
        count += 1
        
    return clusters
    
# clustering functions end

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
#    points = 100*points
    
    points2 = points + 3*np.array([np.array([1, 0]) for i in range(len(points))]) + 3*np.array([np.array([0, 1]) for i in range(len(points))])
    
    # Delaunay
    tri = Delaunay(points)
    tri2 = Delaunay(points2)
    
    r_max = float('-inf')
    r_min = float('inf')
    
    a_shape = AlphaGroups(tri)
    ac = AlphaCluster(a_shape)
    k = 7
    equalize = True
    km = KMeans(k, points, equal_size=equalize)
    if equalize:
        clusters = grow_clusters_from_means(k, a_shape, km.equalized_means)
    else:            
        clusters = grow_clusters_from_means(k, a_shape, km.means)
    
    for i in range(k):
        print("cluster {} size = {}".format(i, len(clusters[i].simplex_indices)))
    
    # fill test
    fill = np.array([[0,0], [3,0], [1.5,3]])#, [0,0]])
    fill = np.array([[0,0], [1,0], [1,1], [2,1], [2,0], [3,0], [3,3], [2,3], [2,2], [1,2], [1,3], [0,3]])
    
    # Plot
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta']
    alpha_complex_filtered = np.array([ simplex for simplex in a_shape.alpha_complex if np.sum(simplex) > -3 ])
    plt.triplot(points[:,0], points[:,1], alpha_complex_filtered, color = 'gray')
    # draw full clusters
    for i in range(k):
        alpha_complex = np.array([ a_shape.alpha_complex[j] for j in clusters[i].simplex_indices ])
        plt.triplot(points[:,0], points[:,1], alpha_complex, color = colors[i])
    # draw only boundaries 
    for i in range(k):
        boundary = clusters[i].get_boundary()
        alpha_complex = np.array([ a_shape.alpha_complex[j] for j in boundary ])
        plt.triplot(points[:,0], points[:,1], alpha_complex, color = colors[i])
    
    km_points = km.equalized_cluster_points
    km_points = km.cluster_points
    # plot kmeans
#    for i in range(k):
#            alpha_shape = AlphaShape(Delaunay(km_points[i]))
#            plt.triplot(km_points[i][:,0], km_points[i][:,1], alpha_shape.boundary_simplices, color = colors[i])

        
    plt.show()