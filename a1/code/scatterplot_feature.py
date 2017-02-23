import pylab as plt
import numpy as np

"""nebulaM nebulam12@gmail.com"""

"""Precondition:dataArray contains data for features; titleList are in titles
   @param dataArray: data(feature) for title
   @param titles: title(city,name etc) for each column in dataArray
   @param features: list of title corresponding to data to plot"""
def scatterplot(dataArray,titles, titleList):
    dim=dataArray.T.shape[1]
    """http://stackoverflow.com/questions/17411940/matplotlib-scatter-plot-legend"""
    colors=['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']
    lenColors=len(colors)
    toPlot=[]
    count=0
    for title in titleList:
        thisMarker='o' if count%2 is 0 else 'x'
        toPlot.append(plt.scatter(np.arange(dim),dataArray.T[np.where(titles == title )],marker='o',color=colors[count%lenColors]))
        count=count+1
    plt.legend((toPlot),(titleList),scatterpoints=1,loc='upper right',ncol=2, fontsize=10)
    plt.show()
