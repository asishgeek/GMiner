#!/usr/bin/python
import sys
import operator
import igraph
from igraph import Graph

MIN_FREQUENCY = 1

def main():
  g = Graph([(0,1), (0,2), (2,3), (3,4), (4,2), (2,5), (5,0), (6,3), (5,6)]) #Test graph
  # Create vertex labels
  vertex_lables = ["A", "B", "A", "C", "C", "D", "D"]
  # Create edge labels
  edge_labels = ["a", "a", "a", "a", "b", "b", "b", "c", "d"]
  g.vs["l"] = vertex_lables
  g.es["l"] = edge_labels
  
  plotGraph(g, g.vs["l"], g.es["l"])
  preprocessGraph(g)
  plotGraph(g, g.vs["nl"], g.es["nl"])

def preprocessGraph(g):
  ''' Delete infrequent vertices and edges. Create new labels based on descending frequency. '''
  #Delete infrequent vertices/edges. Returns vertex and edge frequencies
  (vertexCount, edgeCount) = deleteInfrequentVerticesAndEdges(g)
  
  #Sort the vertex and edge counts in descending order
  vertexCount = sorted(vertexCount, key=operator.itemgetter(1), reverse=True)
  edgeCount = sorted(edgeCount, key=operator.itemgetter(1), reverse=True)
  
  #Re-label graph according to descending order of vertex and edge frequencies 
  relabelGraph(g, vertexCount, edgeCount)

def deleteInfrequentVerticesAndEdges(g):
  '''Delete infrequent vertices and edges.'''
  vertexCount = {}
  edgeCount = {}
  for vertex in g.vs():
    label = vertex["l"]
    vertexCount[label] = vertexCount.get(label, 0) + 1

  for edge in g.es():
    label = edge["l"]
    edgeCount[label] = edgeCount.get(label, 0) + 1
  
  sortedVertexCounts = sorted(vertexCount.iteritems(), key=lambda (x,y): y) #Sorts the map by values.
  sortedEdgeCounts = sorted(edgeCount.iteritems(), key=lambda (x,y): y) #Sorts the map by values.
  verticesToRemove = filter(lambda (x,y): y <= MIN_FREQUENCY, sortedVertexCounts)
  edgesToRemove = filter(lambda (x,y): y<= MIN_FREQUENCY, sortedEdgeCounts)
  
  labels = map(operator.itemgetter(0), verticesToRemove) #Get the first component from each tuple from list of tuples
  #Delete vertices with frequency less than MIN_FREQUENCY
  for label in labels:
    g.vs(l_eq = label).delete()
  labels = map(operator.itemgetter(0), edgesToRemove) #Get the first component from each tuple from list of tuples
  #Delete edges with frequency less than MIN_FREQUENCY
  for label in labels:
    g.es(l_eq = label).delete()
  vertexCount = filter(lambda (x,y): y > MIN_FREQUENCY, vertexCount.iteritems())
  edgeCount = filter(lambda (x,y): y > MIN_FREQUENCY, edgeCount.iteritems())
  return (vertexCount, edgeCount)

def relabelGraph(g, vertexCount, edgeCount):
  ''' The way the new labels are generated as follows:
    Vertices are labeled as V0, V1, ... and edges are labeled as E0, E1, ...
    The new labels are consistent with the old labels i.e. vertices that had the same label continue to have 
    the same label.
  '''
  newLabel = {}
  count = 0
  for (label, freq) in vertexCount:
    for v in g.vs(l_eq=label):
      v["nl"] = "V%d"%count #nl stands for new label
    count += 1

  newLabel = {}
  count = 0
  for (label, freq) in edgeCount:
    for e in g.es(l_eq=label):
      e["nl"] = "E%d"%count #nl stands for new label
    count += 1
 
def plotGraph(g, vertexLabel, edgeLabel):
  layout = g.layout_kamada_kawai()
  style = {}
  style["vertex_label"] = vertexLabel
  style["edge_label"] = edgeLabel
  style["layout"] = layout
  igraph.plot(g, **style)

if __name__ == '__main__':
  sys.exit(main())
