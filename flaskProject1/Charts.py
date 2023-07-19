import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from plotly import offline


class DataCharts:

    def __init__(self) -> None:
        super().__init__()
        self.x = None
        self.y = None
        self.z = None
        self.labels = None

    def GetCoordinate(self, data, labels):
        self.x = []
        self.y = []
        self.z = []
        self.labels = []

        for point in data:
            self.x.append(point[0])
            self.y.append(point[1])
            self.z.append(point[2])
            if labels[data.index(point)] == 0:
                self.labels.append('#5470c6')
            else:
                self.labels.append('#91cc75')

    def ScatterPlot3D(self, coordinate, category):
        # print(self.x)
        self.GetCoordinate(coordinate, category)
        # print(self.x)

        data = [go.Scatter3d(x=self.x, y=self.y, z=self.z, mode='markers', marker=dict(size=12, color=self.labels))]

        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(nticks=4),
                yaxis=dict(nticks=4),
                zaxis=dict(nticks=4)
            )
        )
        fig = go.Figure(data=data, layout=layout)
        offline.plot(fig, filename='static/Pictures/3d-scatter.html', auto_open=True)

        return 0

    def ScatterPlot2D(self, x, y, labels):
        self.x = x
        self.y = y
        self.labels = labels
        return 0


if __name__ == '__main__':
    print("begin...")
    data=[[1,2,3,4],[1,2,3,4],[2,3,4,5]]
    labels=[1,1,0,0]
    pic=DataCharts()
    pic.ScatterPlot3D(data,labels)
