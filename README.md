# Reveal Estate

I worked with Denver based company [Privy](http://www.getprivynow.com) on this project. They introduced me to their business model and need for a clustering algorithm.
The goal of this project is to algorithmically group houses into 'nests' of relevant comps. By taking the purely geographically based zones that Privy already made, and finding the patterns that they used, my algorithm can categorize new data. It's even capable of getting data from a totally new city, with different streets and features and create house nests from scratch.

## Table of Contents
1. [Dataset](#dataset)
2. [Distance Metric](#mri-background)
    * [MRI Pre-Processing](#mri-pre-processing)
3. [Scoring Function](#high-grade-gliomas)
4. [Agglomerative Clustering](#convolutional-neural-networks)
    * [Model Architecture](#model-architecture)
5. [Results](#future-directions)

## Dataset

The data was collected by Privy from multiple listing services (MLS) online. The organization of this data is a big part of the service that they provide. I took their data from the greater DC/Northern Virginia/Maryland area that was already classified manually. After removing points with missing data, there were about 250,000 homes in the data. My goal was to emulate the geographic clusters, and make them even more homogenous by bringing in more features from the MLS dataset.

## Distance Metric

The first thing I needed for the project once I had clean data was a distnace metric to quantify how 'far apart' two houses in the data are. At it's simplest level, this is just the distance 'as the crow flies.' As I added more MLS features to the data, the new distance metric could reward and penalize houses for similar and different features.
This quantity was used in two different places in the project.

* Score clusters that have been made. See [Scoring Function](#Scoring Function).
* Create new clusters. See [Agglomerative Clustering](#Agglomerative Clustering).

## Scoring Function

I made a cluster evaluation metric to asses the quality of a given set of clusters. A [Silhouette Score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) rewards houses that have a small mean distance to other houses in their cluster. It also rewards houses with a large mean distance to other cluster centers.

## Clustering

The first thing my model does is segments the data into all the different ZIP Codes represented. A distance between two houses only exists is they are in the same zip code.
My hierarchical clustering algorithm starts by taking the two 'closest' houses in a ZIP Code and aggregating them into the same cluster. It dose this again and again until there are the same number of clusters in a given Zip Code as Privy started with.

Once my model was tuned and had classified the houses, I had three sets of clusters over the same data to compare.

### 1. Privy zones
* Pros
    * Proven to work well in the real world
    * Easily interpretable
    * No cluster overlap
    * Balanced clusters


* Cons
    * Takes an extremely long time to manually draw zones
    * Okay [silhouette score](#Scoring Function)

### 2. My Zones (Physical Distance)
* Pros
    * Fast algorithmic solution
    * Easily interpretable
    * No cluster overlap


* Cons
    * Cluster imbalance

### 3. My Zones (Other Features Included)
* Pros
    * Strongest [silhouette score](#Scoring Function)
    * Fast algorithmic solution


* Cons
    * Takes an extremely long time to manually draw zones
    * Not as easily interpretable
    * Some cluster overlap

## Results
