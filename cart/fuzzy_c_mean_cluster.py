import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("C:\\Users\David\Documents\CS445\GMM_data_fall2019.txt")
k = 7
m = 1.5
data_len = data.__len__()
weights = np.random.random((k,data_len))
centroids = [None]*k

#Calculates the centroids for each cluster
#Finds the centroid's x and y value for the cluster
#Claculates this by summing up the x or y value for the point by the corresponding weight
#Then divides this value by the total probabilities for the row corresponding to the centroid
def calc_centroids(data):
    data_len = data.__len__()
    for i in range(0,k):
        x_tot = 0
        y_tot = 0
        tot_prob = 0
        for j in range(0,data_len):
            x_tot += weights[i][j] * data[j][0]
            y_tot += weights[i][j] * data[j][1]
            tot_prob += weights[i][j]
        x_cent = x_tot/tot_prob
        y_cent = y_tot/tot_prob
        centroids[i] = [x_cent, y_cent]
    return centroids

#Creates a 2-D array that holds the distance from the centroid (row)
#From the distance of the point(column)
#Returns the 2-D array with the distance from each centroid for each point
def calc_distance(centroids, data):
    data_len = data.__len__()
    distance = np.zeros((k, data_len))
    for i in range(0,k):
        for j in range(0,data_len):
            distance[i][j] = np.linalg.norm(centroids[i] - data[j])
    return distance

#Calculates the weights for each a data_point
#Takes in a point to evaluate
#Finds the sum of the distances for each cluster for the desired point
#Calculates the weight value for given point
def calc_weight(dist_val, col_index, distance):
    dist_sum = 0
    for i in range(0,k):
        dist_sum += (dist_val*dist_val) / (distance[i][col_index] ** distance[i][col_index])
    update = 1/(dist_sum ** (2/m-1))
    return update

#Updates the weights for each data point
#Uses calc_weights function to find the weight for each point
#Returns the update weights
def update_weights(weights,distance,data_len):
    for i in range(0,k):
        for j in range(0,data_len):
            dist_val = distance[i][j]
            weights[i][j] = calc_weight(dist_val,j,distance)
    return weights

#Checks if the weights have changed between iterations
#Checks each point from the old weights to the new weigths
#If there is any difference greater than .01 then return False
#If there is no major change in the weights return True
def check_weights(weights, new_weights,data_len):
    done = True
    for i in range (0,k):
        for j in range(0,data_len):
            if weights[i][j] - new_weights[i][j] >= 0.01:
                done = False
    return done

done = False
count = 0

#Execute Fuzzy C Means
#Runs until weights don't change or has gone through 100 iterations
while not done and count <= 100:
    centroids = calc_centroids(data)
    distance = calc_distance(centroids,data)
    new_weights = update_weights(weights,distance,data_len)
    done = check_weights(weights, new_weights, data_len)
    count += 1
    if not done:
        weights = new_weights

#Handle Plotting
colors = ['r','b','g','y','k','c','m']
#assign each data point to the correct cluster by color for the plot

for j in range(0,data_len):
    max_index = np.argmax(weights[:,j], axis=0)
    plt.scatter(data[j][0],data[j][1],15,c=colors[max_index])

plt.show()


