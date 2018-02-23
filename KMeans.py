#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:15:21 2018

@author: Joel
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 05:17:32 2017

@author: Joel Feske
"""

import matplotlib.pyplot as plt
import numpy as np
from operator import itemgetter
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import ConvexHull
from time import time

# for alpha shape
from AlphaShape import AlphaShape
from scipy.spatial import Delaunay

class KMeans:
    
    '''

    params: optional parameters in brackets 
        k - int - number of means
        data - 2D numpy array of floats - data to be clustered
        [max_iter] - int - maximum number of iterations for the k-means algorithm 
        [equal_size] - bool - whether to equalized cluster size after means are found
        [distance_metric] - str - norm to be used in distance calculations
            - 'Manhattan', 'manhattan' - Manhattan or taxicab norm
            - 'Euclidean', 'euclidean' - standard euclidean distance
            - 'p-norm' - https://en.wikipedia.org/wiki/Norm_(mathematics)#p-norm
                - p = 1 is the Manhattan norm
                - p = 2 is the Euclidean norm
        [p] - float - determines the p-norm to be used if 'p-norm' is selected
        
        
        Note: Currently there is not a way to guarantee that the equalized clusters
              are contiguous. See the Delaunay_Groups.AlphaCluster class which
              partially addresses this problem.

    '''
    
    def __init__(self, k, data, max_iter = 300, equal_size = False, distance_metric = "Euclidian", p = 3):
        self.k = k
        self.data = data
        self.max_iter = max_iter
        self.equal_size = equal_size
        self.distance = self.select_distance_function(distance_metric)
        self.p = p
        
        self.start_means = time()
        self.means = self.iterate()
        self.end_means = time()
        self.means_time = self.end_means - self.start_means
        print("Found means in {}s".format(self.means_time))
        
        self.closest_means = self.find_closest_means(self.means)
        self.closest_points = self.find_closest_points(self.means)
        self.cluster_dicts = self.cluster_dicts()
        
        # numpy array of numpy arrays where self.cluster_points[k] is the points
        # associated with the k_th mean
        self.cluster_points = self.cluster_points()
        
        if self.equal_size:
            
            self.start_equalize = time()
            self.data_dicts = self.equalize()
            self.end_equalize = time()
            self.equalize_time = self.end_equalize - self.start_equalize
            print("Equalized clusters in {}s".format(self.equalize_time))
            
            self.equalized_cluster_dicts = self.equalized_cluster_dicts()
            
            # numpy array of numpy arrays where self.equalized_cluster_points[k] 
            # is the points associated with the k_th mean
            self.equalized_cluster_points = self.equalized_cluster_points()
            # centroids of the equalized clusters
            self.equalized_means = self.equalized_means()
    
    def initialize_means(self, k, data): # should eventually make this use kmeans++
        init_centroid_indices = np.floor(np.linspace(0, len(data), k+1))
        init_means = []
        for i in range(k):
            init_means.append(data[int(init_centroid_indices[i])])
        return np.array(init_means)

    def centroid(self, data):
        data = np.array(data)
        return np.array([np.average(data[:,0]), np.average(data[:,1])])
    
    def select_distance_function(self, tag):
        
        def euclidian_distance(a, b):
            return (a[0] - b[0])**2 + (a[1] - b[1])**2
        
        def manhattan_distance(a, b):
            return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])
        
        # p = 1 is the Manattan norm
        # p = 2 is the Euclidian norm
        def p_norm(a, b):
            return (np.abs(a[0] - b[0])**self.p + np.abs(a[1] - b[1])**self.p)**(1/self.p)
        
        if tag == "Euclidian" or tag == "euclidian":
            return euclidian_distance
        if tag == "Manhattan" or tag == "manhattan":
            return manhattan_distance
        if tag == "p-norm":
            return p_norm
        else:
            return euclidian_distance

    '''
    returns a list of lists of tuples (mean number, squared distance to that mean)
    for each point in data. Each list of tuples is sorted by the squared 
    distance to its mean, e.g.:
    [(2, 0.5), (1, 0.6), (3, 1.1), (5, 1.4), (6, 2.1)]
    ^^^ Here, 2 is the closest mean with a distance of .25, 1 is the second 
    closest, etc. 
    
    '''
    def find_closest_means(self, means):
        closest_means = []
        for i in range(len(self.data)):
            # list of tuples with (mean number, squared distance to that mean)
            # for each datum
            means_for_this_datum = []
            for j in range(self.k):
                means_for_this_datum.append((j, self.distance(self.data[i], means[j])))
            # sort by squared distance to each mean
            means_for_this_datum = sorted(means_for_this_datum, key=itemgetter(1))
            closest_means.append(np.array(means_for_this_datum))
        return np.array(closest_means)
    
    '''
    returns a dictionary with the mean numbers as keys and lists of tuples
    (data index, squared_distance) as values, sorted by squared_distance
    
    '''
    def find_closest_points(self, means):
        closest_points = {}
        for i in range(self.k):
            data_list = []
            for j in range(len(self.data)):
                data_list.append((j, self.distance(self.data[j], means[i])))
            data_list = sorted(data_list, key=itemgetter(1))
            closest_points[i] = np.array(data_list)
        return closest_points
    
    def recalc_means(self, means):
        closest_means = self.find_closest_means(means)
        new_means = []
        for j in range(self.k):
            data_for_this_mean = []
            for i in range(len(self.data)):
                if closest_means[i][0][0] == j:
                    data_for_this_mean.append(self.data[i])
            new_means.append(self.centroid(data_for_this_mean))
        return np.array(new_means)
    
    def iterate(self):
        print("\nFINDING MEANS")
        new_means = self.initialize_means(self.k, self.data)
        count = 0
        total_squared_error = float('inf')
        while(total_squared_error > 0.001*self.k*len(self.data) and count < self.max_iter):
            old_means = new_means
            new_means = self.recalc_means(old_means)
            total_squared_error = np.sum(np.square(new_means - old_means))
            count += 1
        print("{} iterations".format(count))
        return new_means
    
    '''
    returns a list with a dictionary for each point in data
    
    '''
    def init_data_dicts(self):
        closest_means = self.find_closest_means(self.means)
        data_dicts = []
        for i in range(len(self.data)):
            data_dicts.append({'x': self.data[i][0],
                               'y': self.data[i][1],
                               'index':           0,
                               'locked':      False,
                               'closest_means':       closest_means[i],
                               'closest_mean':        closest_means[i][0][0]})
        return data_dicts
    
    def init_mean_dicts(self):
        closest_points = self.find_closest_points(self.means)
        mean_dicts = []
        for i in range(self.k):
            mean_dicts.append({'x': self.means[i][0],
                               'y': self.means[i][1],
                               'index':            0,
                               'closest_points':      closest_points[i],
                               'closest_point':       closest_points[i][0][0]})
        return mean_dicts
    
    # sorted by popularity
    def mean_popularity(self, data_dicts):
        popular_means = []
        for i in range(self.k):
            popular_means.append([i, 0])
        for i in range(len(data_dicts)):
            popular_means[int(data_dicts[i]['closest_mean'])][1] += 1
        popular_means = sorted(popular_means, key=itemgetter(1))
        return np.array(popular_means)
    
    # sorted by mean index
    def mean_popularity_by_mean(self, data_dicts):
        popular_means = []
        for i in range(self.k):
            popular_means.append([i, 0])
        for i in range(len(data_dicts)):
            popular_means[int(data_dicts[i]['closest_mean'])][1] += 1
        return np.array(popular_means)
    
    '''
    (0) Calculate mean popularity
    (1) Start with the mean with the fewest points (call this n_fewest).
    (2) Lock all the points already associated with that mean.
    (3) Move the next (cluster_size - n_fewest) points to that mean and lock them
    
    (4) Recalculate mean popularity
    
    (5) For the mean with the fewest points, repeat (1) - (3)
    
    returns data_dicts with each point's dictionary updated by equalization
    
    '''
    def equalize(self):
        print("\nEQUALIZING CLUSTERS")
        cluster_size = len(self.data)//self.k
        data_dicts = self.init_data_dicts()
        
        # sort means by popularity, most popular first
        popular_means = self.mean_popularity(data_dicts)
        least_popular_mean = popular_means[0][0]
        n_fewest = popular_means[0][1]
        # not actually an average, but an average of the smallest and largest
        average_cluster_size = (popular_means[0][1] + popular_means[-1][1])/2
        percent_difference = (popular_means[-1][1] - popular_means[0][1])/popular_means[0][1]
        
        mean_dicts = self.init_mean_dicts()
        
        count = 0
        while percent_difference > 0.05:
#            if count % 10000 == 0:
#            print("= = = = = = = = = = = = = = = = = = = = =")
#            print("cluster_size = {}".format(cluster_size))
#            print("popular_means[0][1] = {}".format(popular_means[0][1]))
#            print("average_cluster_size  = {}".format(average_cluster_size))
#            print("percent_difference = {}".format(percent_difference))
#            print(popular_means)
            for i in range(len(self.data)):
                if data_dicts[i]['locked'] == False and data_dicts[i]['closest_mean'] == least_popular_mean:
                        data_dicts[i]['locked'] = True
            
            closest_points = mean_dicts[least_popular_mean]['closest_points']            
            
            for i in range(len(self.data)):
                data_index = int(closest_points[i][0])
                if data_dicts[data_index]['closest_mean'] != least_popular_mean \
                and not data_dicts[data_index]['locked'] \
                and n_fewest <= cluster_size:
                    data_dicts[data_index]['closest_mean'] = least_popular_mean
                    data_dicts[data_index]['locked'] = True
                    n_fewest += 1
                    
            popular_means = self.mean_popularity(data_dicts)
            least_popular_mean = popular_means[0][0]
            n_fewest = popular_means[0][1]
            average_cluster_size = (popular_means[0][1] + popular_means[-1][1])/2
            percent_difference = (popular_means[-1][1] - popular_means[0][1])/popular_means[0][1]
            count += 1
            if count == 100:
                print("\nWarning: terminated early")
                print("Largest cluster is {} percent larger than smallest cluster\n".format('{0:.2f}'.format(100*percent_difference)))
                break
        
        print("\n= = = = = = = = = = = = = = = = = = = = =")
        print("cluster_size = {}".format(cluster_size))
        print("popular_means[0][1] = {}".format(popular_means[0][1]))
        print("average_cluster_size  = {}".format(average_cluster_size))
        print("percent_difference = {}".format(percent_difference))
        print(popular_means)
        print()
        
        print("{} iterations".format(count))
        return data_dicts            
    
    
    # ===
        '''
    (0) Calculate mean popularity
    (1) Start with the mean with the fewest points (call this n_fewest).
    (2) Lock all the points already associated with that mean.
    (3) Move the next (cluster_size - n_fewest) points to that mean and lock them
    
    (4) Recalculate mean popularity
    
    (5) For the mean with the fewest points, repeat (1) - (3)
    
    returns data_dicts with each point's dictionary updated by equalization
    
    '''
    def equalize_2(self):
        print("\nEQUALIZING CLUSTERS")
        cluster_size = len(self.data)//self.k
        data_dicts = self.init_data_dicts()
        
        popular_means = self.mean_popularity_by_mean(data_dicts)

        percent_difference = (popular_means[-1][1] - popular_means[0][1])/popular_means[0][1]
        
        mean_dicts = self.init_mean_dicts()
        
        count = 0
        for i in range(self.k):
            # if the i_th mean has fewer than 'cluster_size' points
            if popular_means[i][1] < cluster_size:
                # lock the points it does have
                for j in range(len(self.data)):
                    if data_dicts[j]['closest_mean'] == i and not data_dicts[j]['locked']:
                        data_dicts[j]['locked'] = True
                # top it off with nearest unlocked points
                
            
        
        print("\n= = = = = = = = = = = = = = = = = = = = =")
        print("cluster_size = {}".format(cluster_size))
        print("popular_means[0][1] = {}".format(popular_means[0][1]))
        print("percent_difference = {}".format(percent_difference))
        print(popular_means)
        print()
        
        print("{} iterations".format(count))
        return data_dicts            
    # ===
    
    def cluster_dicts(self):
        data_dicts = self.init_data_dicts()
        clusters = []
        for i in range(self.k):
            clusters.append(np.array([x for x in data_dicts if x['closest_mean'] == i]))
        return np.array(clusters)
    
    def equalized_cluster_dicts(self):
        equalized_clusters = []
        for i in range(self.k):
            equalized_clusters.append(np.array([x for x in self.data_dicts if x['closest_mean'] == i]))
        return np.array(equalized_clusters)
    
    def cluster_points(self):
        cluster_points = []
        for i in range(self.k):
            cluster_points.append([])
            for j in range(len(self.cluster_dicts[i])):
                cluster_points[i].append(np.array([self.cluster_dicts[i][j]['x'], self.cluster_dicts[i][j]['y']]))
            cluster_points[i] = np.array(cluster_points[i])
        return np.array(cluster_points)
    
    def equalized_cluster_points(self):
        equalized_cluster_points = []
        for i in range(self.k):
            equalized_cluster_points.append([])
            for j in range(len(self.equalized_cluster_dicts[i])):
                equalized_cluster_points[i].append(np.array([self.equalized_cluster_dicts[i][j]['x'], self.equalized_cluster_dicts[i][j]['y']]))
            equalized_cluster_points[i] = np.array(equalized_cluster_points[i])
        return np.array(equalized_cluster_points)
    
    def equalized_means(self):
        means = np.zeros((self.k, 2))
        for i in range(self.k):
            means[i] = self.centroid(np.array([(e[0], e[1]) for e in self.equalized_cluster_points[i]]))
        return means

if __name__ == "__main__":
    print("KMeans")
    k = 6
    n = 25
    max_iter = 300
    equal_size = True
    distance_metric = "euclidean"
    p = 3
    
    # Change 'case' to select different data distributions
    case = 0
    if case == 0:
        data = np.random.randint(1,100,(10000,2))
    elif case == 1:
        data1 = np.random.randint(1,50,(100,2))
        data2 = np.random.randint(51,100,(5000,2))
        data = np.concatenate((data1, data2), axis=0)
    elif case == 2:
        data1 = np.random.randint(1,25,(n,2))
        data2 = np.random.randint(26,50,(2*n,2))
        data3 = np.random.randint(51,75,(3*n,2))
        data4 = np.random.randint(76,100,(4*n,2))
        data = np.concatenate((data1, data2, data3, data4), axis=0)
    elif case == 3:
        mu, sigma = 3., 1.1 # mean and standard deviation
        data = np.random.lognormal(mu, sigma, (10000,2))
    elif case == 4:
        mu, sigma = 10, 1
        data = np.random.normal(mu, sigma, (10000,2))
        data = np.square(data)
    
    # KMeans
    km = KMeans(k, data, max_iter, equal_size, distance_metric, p)
    
    print()
    for i in range(k):
        print("km.clusters[{}].shape = {}".format(i, km.cluster_dicts[i].shape))

    if equal_size:
        print()
        for i in range(k):
            print("km.equalized_clusters[{}].shape = {}".format(i, km.equalized_cluster_dicts[i].shape))
        
    plot_Voronoi = False
    plot_hulls = False
    plot_alpha = True
    plot_points = False
    colors = ['r','g','b','c','m','y','orange']

    rr = np.random.random()
    rg = np.random.random()
    rb = np.random.random()
    colors = []
    for i in range(k):
        colors.append(((rr*(i+1))%1, (rg*(i+1))%1, (rb*(i+1))%1,))
        
    colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta', 'black']
        
    # Figure 1 -- standard k-means
    cluster_points = km.cluster_points
    points = km.cluster_points
    
    if plot_Voronoi:
        vor1 = Voronoi(km.means)
        voronoi_plot_2d(vor1)
    
    if plot_hulls:
        hulls1 = []
        for i in range(k):
            if len(points[i]) > 2:
                hulls1.append(ConvexHull(points[i]))
            else: # if you cannot make a simplex
                plt.plot(points[i][:,0], points[i][:,1], 'k-')
                
        for hull in hulls1:
            for simplex in hull.simplices:
                plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'k-')
    
    if plot_alpha:
        for i in range(k):
            a_shape = AlphaShape(Delaunay(points[i]))
            plt.triplot(points[i][:,0], points[i][:,1], a_shape.boundary_simplices, color = colors[i])

    if plot_points:
        for i in range(k):
            plt.scatter(points[i][:,0], points[i][:,1], marker = '.', color = colors[i])
            plt.scatter(km.means[i][0], km.means[i][1], marker = '*', color = 'k')
    
    margin = 10
    x_min = np.min(data[:,0]) - margin
    x_max = np.max(data[:,0]) + margin
    y_min = np.min(data[:,1]) - margin
    y_max = np.max(data[:,1]) + margin
    plt.axis([x_min, x_max, y_min, y_max])
    plt.show()
    
    
    # Figure 2 -- equal size k-means
    if equal_size:
        equalized_cluster_points = km.equalized_cluster_points
        points = km.equalized_cluster_points
        
        if plot_Voronoi:
            vor2 = Voronoi(km.equalized_means)
            voronoi_plot_2d(vor2)
        
        if plot_hulls:
            hulls2 = []
            for i in range(k):
                if len(points[i]) > 2:
                    hulls2.append(ConvexHull(points[i]))
                else: # if you cannot make a simplex
                    plt.plot(points[i][:,0], points[i][:,1], 'k-')
                    
            for hull in hulls2:
                for simplex in hull.simplices:
                    plt.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'k-')
        
        if plot_alpha:
            for i in range(k):
                a_shape = AlphaShape(Delaunay(points[i]))
                plt.triplot(points[i][:,0], points[i][:,1], a_shape.boundary_simplices, color = colors[i])
        
        if plot_points:
            for i in range(k):
                plt.scatter(points[i][:,0], points[i][:,1], marker = '.', color = colors[i])
                plt.scatter(km.equalized_means[i][0], km.equalized_means[i][1], marker = '*', color = 'k')
    
    margin = 10
    x_min = np.min(data[:,0]) - margin
    x_max = np.max(data[:,0]) + margin
    y_min = np.min(data[:,1]) - margin
    y_max = np.max(data[:,1]) + margin
    plt.axis([x_min, x_max, y_min, y_max])
    plt.show()