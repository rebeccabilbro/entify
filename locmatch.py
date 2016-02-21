#!/usr/bin/python
# locmatch.py

# Title:        Location Matcher
# Version:      1.0
# Author:       Rebecca Bilbro
# Date:         2/21/16
# Organization: District Data Labs

"""
ALGORITHM:
Defines entities using kmeans clustering to identify locations based
latitude & longitude.

ABOUT THE DATA:
Average Daily Traffic (ADT) counts are analogous to a census count of
vehicles on city streets. These counts provide a close approximation
to the actual number of vehicles passing through a given location on
an average weekday. Since it is not possible to count every vehicle on
every city street, sample counts are taken along larger streets to get
an estimate of traffic on half-mile or one-mile street segments. ADT
counts are used by city planners, transportation engineers, real-estate
developers, marketers and many others for myriad planning and operational
purposes. Data Owner: Transportation. Time Period: 2006. Frequency: A
citywide count is taken approximately every 10 years.

SOURCE + METADATA:
http://catalog.data.gov/dataset/average-daily-traffic-counts-3968f
"""

######################################################################
# IMPORTS
######################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

######################################################################
# ENTITY RESOLUTION - FILE ANNOTATION METHOD
######################################################################
def entify(incsv,outcsv,outimg):
    """
    Takes as input a csv file with longitude and latitude data,
    performs kmeans clustering and assigns labels according to cluster
    number.

    Outputs updated file with entity cluster labels and visualization
    of the resultant clusters.
    """
    N_CLUSTERS = 50
    df = pd.read_csv(incsv)
    lldata = df[["Latitude","Longitude"]].values

    model = KMeans(n_clusters=N_CLUSTERS)
    model.fit(lldata)
    labels = model.labels_
    centroids = model.cluster_centers_
    df["Entity Label"] = labels
    df.to_csv(outcsv)

    for i in range(N_CLUSTERS):
        datapoints = lldata[np.where(labels==i)]
        plt.plot(datapoints[:,0],datapoints[:,1],'k.')
        centers = plt.plot(centroids[i,0],centroids[i,1],'x')
        plt.setp(centers,markersize=20.0)
        plt.setp(centers,markeredgewidth=2.0)
    plt.savefig(outimg)


if __name__ == '__main__':
    INFILE    = "Average_Daily_Traffic_Counts.csv"
    OUTFILE   = "Average_Daily_Traffic_Counts_labelled.csv"
    IMAGEPATH = "clustersv1.png"
    entify(INFILE,OUTFILE,IMAGEPATH)
