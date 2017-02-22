import pylab as plt
import numpy as np

"""nebulaM nebulam12@gmail.com"""

"""Precondition:dataArray contains data for features; features are in titles
   @param dataArray: data of feature
   @param titles: title(feature name) for each column in dataArray
   @param features: list of feature to plot"""
def scatterplot(dataArray,titles, features):
    dim=dataArray.T.shape[1]
    """http://stackoverflow.com/questions/17411940/matplotlib-scatter-plot-legend"""
    colors=['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    lenColors=len(colors)
    toPlot=[]
    count=0
    for feature in features:
        thisMarker='o' if count%2 is 0 else 'x'
        toPlot.append(plt.scatter(np.arange(dim),dataArray.T[np.where(titles == feature )],marker='o',color=colors[count%lenColors]))
        count=count+1
    plt.legend((toPlot),(features),scatterpoints=1,loc='upper right',ncol=2, fontsize=10)
    plt.show()
