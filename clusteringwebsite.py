#import packages
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import hdbscan
from kneed import KneeLocator
import pickle
import os
import time

"""
# World Happiness Clustering Analysis
"""

#functions:

#uses plotly to generate 3d plot of clusters
def get3Dplots(df,x,y,z):
    fig = px.scatter_3d(df, x=x,y = y, z=z, color = 'custom_rank')
    st.plotly_chart(fig)

#uses plotly to generate 3d plot of clusters, highlights country in red
def get3Dplots(df,x_colname,y_colname,z_colname, query_country, color_by='custom_rank'):
# Overloaded function that takes in query country (row) and plots it along with the scatter plot
    scatter = go.Scatter3d(x=df[x_colname], y=df[y_colname], z=df[z_colname], mode='markers', marker=dict(size=3, opacity=0.8, color=df[color_by], colorscale='viridis'))
    query_point = go.Scatter3d(x=[query_country[x_colname]], y=[query_country[y_colname]], z=[query_country[z_colname]], marker=dict(size=5, opacity=1, color='red'))

    layout = go.Layout(
      scene=dict(xaxis_title=x_colname, yaxis_title=y_colname, zaxis_title=z_colname)
    )

    fig = go.Figure(data=[scatter, query_point], layout=layout)
    st.plotly_chart(fig)

#use to get index clustering validation scores
def getscores(df,labels):
    #scores
    db =davies_bouldin_score(df, labels)
    ch =calinski_harabasz_score(df, labels)
    sc = silhouette_score(df, labels)

    #print
    print("Davies-Bouldin index optics:", db)
    print("Calinski-Harabasz index optics:", ch)
    print("Silhouette coefficient optics:", sc)

#turns year string to index in datalist
def getyearindex(year):
    yearsindexlist = ['2022', '2021', '2020', '2019', '2018', '2017', '2016', '2015']
    return yearsindexlist.index(year)

#used to get pca weights for equation
def getpcaweights(currentdata):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(currentdata)

    # Create PCA object with desired number of components
    pca = PCA(n_components=1)

    # Fit PCA model to scaled data
    pca.fit(scaled_data)

    # Transform data into principal components
    principal_components = pca.transform(scaled_data)
    components = pca.components_
    totalcomponents = np.array(components)/float(sum(abs(components[0])))
    return totalcomponents[0]

#generate a dictionary of centroid for every cluster
def getmeanhappiness(dataset, criteria):
    numclusters = df[criteria].nunique()
    print(numclusters)
    grouped = df.groupby(criteria)

    # iterate over the groups and print the data for each group
    dic={}
    for name, group in grouped:

        row_mean = getcentroid(group)
        dic[name] = pc(getpcaweights(dataset),row_mean)

    return dic


#generates pc equation calculation
def pc(weights,row_mean):
    val = 0
    for i in range(len(weights)):
        val+=weights[i]*row_mean[i]
    return val

#gets a cluster centroid
def getcentroid(group):
    row_mean = group.mean(axis=0)
    return row_mean

#turns clusters pre assigned to custom clusters
def get_custom_rank(d):
    sorted_indices = [i[0] for i in sorted(enumerate(d.keys()), key=lambda x: d[x[1]], reverse=True)]
    new_dict = {k : int(sorted_indices[i]) for i, k in enumerate(d)}
    return new_dict

#get centroids by custom rank
def getcentroids(dataset, criteria='custom_rank'):
    numclusters = dataset[criteria].nunique()
    print(f'numclusters = {numclusters}')
    grouped = dataset.groupby(criteria)

    # iterate over the groups and print the data for each group
    centroids = pd.DataFrame()

    for name, group in grouped:
        row_mean = getcentroid(group).to_frame().T
        centroids = pd.concat([centroids, row_mean], ignore_index=True)

    return centroids

#returns the cluster a country should be moving towards as the closest happier cluster
def nextbestone(country,centroidlist): #country is arraylike, both have happiness rank and raw features
    #if country is in cluster 0, it is already happy
    if(country["custom_rank"] == 0):
        return -1

    #fileter by happier clusters
    df = centroidlist.loc[centroidlist["custom_rank"] < country["custom_rank"]]
    df['distance']=pd.Series(dtype=float)

    #calculate distance to each cluster
    for i in range(len(df)):
        df.loc[i,'distance'] = np.linalg.norm( np.array(centroidlist.iloc[i,:6]) - np.array(country.iloc[:6]))

    #return cluster with lowest distance
    return (df.loc[df['distance'].idxmin()])


#sets 3d plots, but line going to next cluster
def get3Dplots_with_line(df, x_colname,y_colname,z_colname, lstart, lend, color_by='custom_rank'):

    scatter = go.Scatter3d(x=df[x_colname], y=df[y_colname], z=df[z_colname], mode='markers', marker=dict(size=3, color=df[color_by], colorscale='viridis' , opacity=0.8))

    #generate line coordinates
    lx = np.array( [lstart[x_colname], lend[x_colname]] )
    ly = np.array( [lstart[y_colname], lend[y_colname]] )
    lz = np.array( [lstart[z_colname], lend[z_colname]] )

    #generate line and arroy
    line = go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='red', width=4))
    arrow = go.Cone(x=[lx[1]], y=[ly[1]], z=[lz[1]], u= [0.5*(lx[1]-lx[0])], v=[0.5*(ly[1]-ly[0])], w=[0.5*(lz[1]-lz[0])], anchor="tip",  hoverinfo="none" ,colorscale = [[0,'red'],[1,'red']])

    source = go.Scatter3d(x=[lx[0]], y=[ly[0]], z=[lz[0]], marker=dict(size=4, opacity=1, color='red'))
    layout = go.Layout(
        scene=dict(xaxis_title=x_colname, yaxis_title=y_colname, zaxis_title=z_colname)
    )

    fig = go.Figure(data=[scatter, line, arrow, source], layout=layout)

    st.plotly_chart(fig)


#shows the movement of countries per year starting from 2015
def countrylinemaker(countryname,datalist, countrymappinglist,x_colname,y_colname,z_colname):

    #start at 2015
    x = datalist[7]
    c1 = x.iloc[countrymappinglist[0][countryname]]
    data1 = []

    layout = go.Layout(
    scene=dict(xaxis_title=x_colname, yaxis_title=y_colname, zaxis_title=z_colname)
    )

    for i in range(len(datalist)-2,0,-1):
        #calculate position of country in next highest year
        x = datalist[i]
        c2 = x.iloc[countrymappinglist[i][countryname]]

        #generate line coordinates
        lx = np.array( [c1[x_colname], c2[x_colname]] )
        ly = np.array( [c1[y_colname], c2[y_colname]] )
        lz = np.array( [c1[z_colname], c2[z_colname]] )

        line = go.Scatter3d(x=lx, y=ly, z=lz, mode='lines', line=dict(color='red', width=3, showscale=False))
        arrow = go.Cone(x=[lx[1]], y=[ly[1]], z=[lz[1]], u= [0.5*(lx[1]-lx[0])], v=[0.5*(ly[1]-ly[0])], w=[0.5*(lz[1]-lz[0])], anchor="tip",  hoverinfo="none" )

        data1.append(line)
        data1.append(arrow)

        c1=c2
    fig = go.Figure(data=data1, layout=layout)

    st.plotly_chart(fig)


def main(desiredcountry,year):
    #fix country syntax
    desiredcountry = desiredcountry.lower().title()
    print(desiredcountry)

    #required data
    yearindex = getyearindex(year)
    currentdata = datalist[yearindex]
    centroidlist = getcentroids(currentdata)

    #get country row and next best cluster
    query_country = currentdata.iloc[countrymappinglist[yearindex][desiredcountry]]
    nextbestcentroid = nextbestone(query_country, centroidlist)


    st.write("This country has a happiness rank of " + str(query_country['custom_rank']))
    #if country is happy, nothing to improve
    if(isinstance(nextbestcentroid, int)):
      print("Nothing")
      st.write("Country is already in the best cluster")

    #if country isn't happy go to next best cluster
    else:
      dis_vector = list(np.array(nextbestcentroid.iloc[:6]) - np.array(query_country.iloc[:6]))
      idx = dis_vector.index(max(dis_vector))
      val = currentdata.columns[idx]
      st.write("We recommend a change in " + str(val) + " to improve!")

    #need to consider the case where nextbestcentroid = -1 (you are already in the best cluster)
    if(query_country["custom_rank"] == 0):
        st.write("You are already in the best cluster")
        get3Dplots(currentdata,currentdata.columns[3],currentdata.columns[1],currentdata.columns[2], query_country=query_country)
        return
    else:
        get3Dplots_with_line(currentdata, 'GDP', 'Health', 'Family', query_country, nextbestcentroid)

#generates lines tracking country across years
def linemaker(desiredcountry2):
    desiredcountry2 = desiredcountry2.lower().title()
    datalist[4].rename(columns={'Healthy': 'Health'},inplace=True)

    #coloumns you make lines for
    x_colname = 'GDP'
    y_colname = 'Health'
    z_colname = 'Family'

    countrylinemaker(desiredcountry2,datalist, countrymappinglist,x_colname,y_colname,z_colname)

#shows the different clusters we attempted
def clusterpresentor(desiredclustering):
    #load pickles
    with open('cluster_pickles/dbscan_clustering.pickle', 'rb') as f:
        kmeans_clustering= pickle.load(f)
    with open('cluster_pickles/kmeans_clustering.pickle', 'rb') as f:
        dbscan_clustering  = pickle.load(f)
    with open('cluster_pickles/kmedoids_clustering.pickle', 'rb') as f:
        kmedoids_clustering = pickle.load(f)
    with open('cluster_pickles/optics_clustering.pickle', 'rb') as f:
        optics_clustering = pickle.load(f)

    #titles
    dbscan_clustering.update_layout(title="dbscan clustering")
    kmeans_clustering.update_layout(title="kmeans clustering")
    kmedoids_clustering.update_layout(title="kmedoids clustering")
    optics_clustering.update_layout(title="optics clustering")

    #plots
    st.plotly_chart(dbscan_clustering)
    st.plotly_chart(kmeans_clustering)
    st.plotly_chart(kmedoids_clustering)
    st.plotly_chart(optics_clustering)


#main

#load important pickles
with open('my_datalist.pickle', 'rb') as f:
    datalist = pickle.load(f)
with open('countrymappinglist.pickle', 'rb') as f:
    countrymappinglist = pickle.load(f)


#form for first demand, generates clusters for a country
with st.form(key='inputcountryform'):
    #countryname and year
    cname = st.text_input("what country?")
    year = str(st.selectbox("what year?", ['2022', '2021', '2020', '2019', '2018', '2017', '2016', '2015']))

    submit = st.form_submit_button(label='Submit')
    if submit:
        desiredcountry = cname
        # process form data here
        st.success("Submitted!")
        main(desiredcountry,year)
        # clear form data
        cname = ""
        year = ""


#for of lines of country
with st.form(key='linecountryform'):
    #country name
    cname = st.text_input("what country do you want the lines for?")

    submit = st.form_submit_button(label='Submit')
    if submit:
        desiredcountry2 = cname
        # process form data here
        st.success("Submitted!")
        linemaker(desiredcountry2)
        # clear form data
        cname = ""


st.write("---")

#form to generate cluster
with st.form(key='clustersform'):
    submit = st.form_submit_button(label='Click to generate clusters')
    if submit:
        desiredclustering = cname
        # process form data here
        clusterpresentor(desiredclustering)
        # clear form data
        cname = ""

st.write("---")


###########################
###########################
###########################
###########################
###########################
###########################
###########################

#code to generate slider the makes figures for each year

#load pickles
with open('KmeansYearlyPickle/kmeansyearlyclusters2015.pickle', 'rb') as f:
    fig2015 = pickle.load(f)
with open('KmeansYearlyPickle/kmeansyearlyclusters2016.pickle', 'rb') as f:
    fig2016 = pickle.load(f)
with open('KmeansYearlyPickle/kmeansyearlyclusters2017.pickle', 'rb') as f:
    fig2017 = pickle.load(f)
with open('KmeansYearlyPickle/kmeansyearlyclusters2018.pickle', 'rb') as f:
    fig2018 = pickle.load(f)
with open('KmeansYearlyPickle/kmeansyearlyclusters2019.pickle', 'rb') as f:
    fig2019 = pickle.load(f)
with open('KmeansYearlyPickle/kmeansyearlyclusters2020.pickle', 'rb') as f:
    fig2020 = pickle.load(f)
with open('KmeansYearlyPickle/kmeansyearlyclusters2021.pickle', 'rb') as f:
    fig2021 = pickle.load(f)
with open('KmeansYearlyPickle/kmeansyearlyclusters2022.pickle', 'rb') as f:
    fig2022 = pickle.load(f)

figures = [fig2015,
            fig2016,
            fig2017,
            fig2018,
            fig2019,
            fig2020,
            fig2021,
            fig2022]

# create a slider widget with a range of 1 to 8 and an initial value of 1
st.session_state.slider_value = st.slider("Select a figure", 2015, 2022)

# display the selected figure
st.write("Selected year:", st.session_state.slider_value )
st.plotly_chart(figures[st.session_state.slider_value - 2015])
time.sleep(0.1)


###########################
###########################
###########################
###########################
###########################
###########################
###########################

st.write("---")
