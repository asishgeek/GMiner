#!/usr/bin/python
import sys
import operator
import igraph
from igraph import Graph
''' GMiner algorithm. http://dl.acm.org/citation.cfm?id=1598339
Original labels of vertices and edges are identified by the attribute "l".
Frequency based labels are identified by the attribute "nl".
'''
MIN_FREQUENCY = 1

def main():
  g = createTestGraph()
  
  plotGraph(g, g.vs["l"], g.es["l"])
  preprocessGraph(g)
  plotGraph(g, g.vs["nl"], g.es["nl"])
  
  ig = createInstanceGraph(g,["V0","E0","V0","E0","V1"])
  label = map(str, range(0,len(ig.vs) - 1))
  plotGraph(ig, label, label)
  
  #Create one edge graphs. S1 contains all the one edge graphs

    
def createInstanceGraph(g, p):
  ''' Creates the instance of graph of pattern p. Where the pattern p is a two edge graph
      denoted by a list: [V0,E0,V1,E1,V2] '''
  ig = Graph() #Instance graph
  for e0 in g.es(nl_eq = p[1]):
    (v0, v1) = getVertices(g,e0)
    if v0["nl"] == p[0] and v1["nl"] == p[2]:
      for v2 in v1.neighbors():
        e1 = getEdge(g, v1, v2)
        if e1["nl"] == p[3] and v2["nl"] == p[4]:
          # Found a pattern
          pid = addPattern(ig, e0, e1)
  
  del g.es["partof"]
  return ig
        
def addPattern(g, e1, e2):
  ''' Adds the pattern to identified by the two edges to the instance graph g.
      In case edges are shared between patterns then connections are created.'''
  g.add_vertex()
  pid = len(g.vs) - 1
  prevPids = []
  for e in [e1, e2]:
    if "partof" in e.attributes() and e["partof"]:
      prevPids.extend(e["partof"])
    else:
      e["partof"] = []

  for prev in set(prevPids):
    g.add_edge(pid,prev)

  e1["partof"].append(pid)
  e2["partof"].append(pid)
  return pid  


def getEdge(g, v1, v2):
  ''' Gets the edge object between the two vertex objects in graph g. '''
  return g.es[g.get_eid(v1.index, v2.index)]

def getVertices(g, e):
  ''' Gets the vertices joined by the edge e '''
  vertices = [e.source, e.target]
  vertices = sorted(vertices, key=lambda(x): g.vs[x]["nl"]) #Sort by lexicographic orider of labels
  return (g.vs[vertices[0]], g.vs[vertices[1]])


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
    The new labels are consistent with the old labels i.e. vertices/edges that had the same label continue to have 
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

def createTestGraph():
  ''' Test code. Creates a graph to test on'''
  #g = Graph([(0,1), (0,2), (2,3), (3,4), (4,2), (2,5), (5,0), (6,3), (5,6)]) #Test graph
  g = Graph([(0,1),(1,2),(0,2),(2,3),(3,4),(4,5),(3,5),(5,6),(3,6),(6,7),(2,7),(7,8),(8,9),(9,10),(8,10),(6,8)])
  # Create vertex labels
  vertex_lables = ["A", "B", "B", "B", "A", "B", "B", "C", "A", "B", "B"]
  # Create edge labels
  #edge_labels = ["a", "a", "a", "a", "b", "b", "b", "c", "d"]
  edge_labels = ["a"]
  g.vs["l"] = vertex_lables
  g.es["l"] = edge_labels
  return g

if __name__ == '__main__':
  sys.exit(main())
