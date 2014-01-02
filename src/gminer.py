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
  #plotGraph(g, g.vs["l"], g.es["l"])
  preprocessGraph(g)
  
  p = Pattern(g, [g.es[0], g.es[1], g.es[2]])
  print p.getDFSCode()
  for cp in p.getChildPatterns():
    print cp.getDFSCode()
  p = p.getChildPatterns()[0]
  for cp in p.getChildPatterns():
    print cp.getDFSCode()


  label = map(lambda (x,y): "%s_%s"%(x,y), zip(g.vs["nl"], g.vs["indices"]))
  plotGraph(g, label, g.es["nl"])

  '''
  ig = createInstanceGraph(g,["V0","E0","V0","E0","V1"])
  newIg = growInstanceGraph(g, ig, ["V1", "V0"])
  label = map(str, range(0,len(newIg.vs) - 1))
  #plotGraph(newIg, [], [])
  print newIg
  
  #Create one edge graphs. S1 contains all the one edge graphs
  '''

def growInstanceGraph(g, ig, e):
  ''' Grow the instance graph ig by adding the edge e. e is a given by labels i.e. ["V0", "V1"]
      Return the new instance graph. '''
  newIg = Graph()
  for v in ig.vs:
    cliqueSize = 0
    for gv in v["G"]:
      if gv["nl"] != e[0]:
        continue

      for gu in gv.neighbors():
        if gu["nl"] != e[1]:
          continue
        print "Found desired edge"
        cliqueSize += 1
        
    nv = addClique(newIg, cliqueSize)
    if nv:
      v["child"] = nv
  
  #Create edges between vertices of newIg
  for eig in ig.es:
    [v0, v1] = getVertices(ig, eig)
    if "chlid" in v0.attributes() and "chlid" in v1.attributes():
      cv0 = v0["child"] #cv0 is a node in newIg
      cv1 = v1["child"] #cv1 is a node in newIg
      newIg.add_edge(cv0, cv1)

  #TODO: delete child attributes
  return newIg

def addClique(g, cliqueSize):
  if cliqueSize <= 0:
    return None
  
  g.add_vertex()
  vid = g.vs.indices[-1] # Id of the last added vertex
  for i in range(1, cliqueSize):
    g.add_vertex()
    for j in range(i-1,-1,-1):
      g.add_edge(vid + j, vid + i)

def createInstanceGraph(g, p):
  ''' Creates the instance of graph of pattern p. Where p is a two edge graph
      denoted by a list: e.g. [V0,E0,V1,E1,V2] '''
  ig = Graph() #Instance graph
  for e0 in g.es(nl_eq = p[1]):
    (v0, v1) = getVertices(g,e0)
    if v0["nl"] == p[0] and v1["nl"] == p[2]:
      for v2 in v1.neighbors():
        e1 = getEdge(g, v1, v2)

        if e1["nl"] == p[3] and v2["nl"] == p[4]:
          # Found a pattern
          pid = addPattern(ig, e0, e1)
          ''' Create connection between node of instance graph and the corresponding nodes 
          of the original graph. e.g. If v is a vertex in the instance graph then 
          G(v) = {u such that u belongs to the vertex set of P} where P is the
          pattern based on which the instance graph has been created. '''
          v_ig = ig.vs[pid]
          v_ig["G"] = [v0, v1, v2]
  
  del g.es["partof"]
  return ig
        
def addPattern(g, e1, e2):
  ''' Adds the pattern identified by the two edges to the instance graph g.
      In case edges are shared between patterns then connections are created.'''
  g.add_vertex()
  pid = len(g.vs) - 1 #pid represent the instance graph node that was added
  prevPids = [] # prevPids for a given edge is a list of other instance graph
                # nodes that the edge is also part of. So that connections
                # can be created between instance graph nodes that share edges.
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
  if g[v1, v2] == 1:
    return g.es[g.get_eid(v1.index, v2.index)]
  else:
    return None

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
  if vertexLabel:
    style["vertex_label"] = vertexLabel
  if edgeLabel:
    style["edge_label"] = edgeLabel
  style["layout"] = layout
  igraph.plot(g, **style)

def createTestGraph():
  ''' Test code. Creates a graph to test on'''
  #g = Graph([(0,1), (0,2), (2,3), (3,4), (4,2), (2,5), (5,0), (6,3), (5,6)]) #Test graph
  g = Graph([(0,1),(1,2),(0,2),(2,3),(3,4),(4,5),(3,5),(5,6),(3,6),(6,7),(2,7),(7,8),(8,9),(9,10),(8,10),(6,8)])
  # Create vertex labels
  vertex_lables = ["A", "B", "B", "B", "A", "B", "B", "C", "A", "B", "B"]
  vertex_indices = map(str, g.vs.indices)
  # Create edge labels
  #edge_labels = ["a", "a", "a", "a", "b", "b", "b", "c", "d"]
  edge_labels = ["a"]
  g.vs["l"] = vertex_lables
  g.vs["indices"] = vertex_indices
  g.es["l"] = edge_labels
  '''
  g = Graph([(0,1), (1,2), (2,0), (2,3)])
  g.vs["nl"] = ["A", "B", "C", "D"]
  g.es["nl"] = ["e"]
  '''
  return g

def computeDFSCode(G, edgeList):
  [source, target] = zip(*([e.tuple for e in edgeList]))
  vertexIndices = set(source + target)
  vertices = G.vs[vertexIndices]
  vertices = (sorted(vertices, key=lambda x: x['nl']))
  vertexSet = set(map(lambda x: x.index, vertices))
  dfsCode = ""
  for v in vertices:
    if "visited" not in v.attributes():
      dfsCode = visit(v, dfsCode)
      dfsCode = dfsVisit(G, v, dfsCode, vertexSet)

  del G.vs["visited"]
  del G.es["forward"]
  return dfsCode

def dfsVisit(G, vertex, dfsCode, vertexSet):
  #print "visiting: %d"%vertex.index
  #print "dfsCode: %s"%dfsCode
  neighbors = sorted(vertex.neighbors(), key=lambda x: (x["nl"], getEdge(G, vertex, x)["nl"])) #Sort by vertex label followed by edge label
  for u in neighbors:
    if u.index not in vertexSet:
      #print "%d not in vertex set"%u.index
      continue
    if "visited" in u.attributes() and u["visited"] == True:
      #print "%d has already been visited"%u.index
      continue

    e = getEdge(G, vertex, u)
    dfsCode = visit(e, dfsCode) 
    dfsCode = visit(u, dfsCode)
    dfsCode = dfsVisit(G, u, dfsCode, vertexSet)

  #Get backward edges and add it in the DFS code
  for u in neighbors:
    if u.index not in vertexSet:
      continue
    e = getEdge(G, vertex, u)
    if "forward" in e.attributes() and e["forward"] == True:
      continue
    dfsCode = visit(e, dfsCode)
    dfsCode += u["nl"]
    
  return dfsCode
      
def visit(item, dfsCode):
  #print "visiting item: %s"%str(item)
  if type(item) == igraph.Edge:
    item["forward"] = True
  elif type(item) == igraph.Vertex:
    item["visited"] = True
  else:
    raise Exception("Invalid type of item should be either a vertex or an edge")
  return (dfsCode + item["nl"])

class Pattern:
  ''' A list of edges repsenting a pattern. The list of edges are sorted according to the DFS order. The DFS code of the
  pattern is also persisted. '''

  def __init__(self, G, edgeList):
    if type(edgeList) != list:
      raise Exception("argument must be a list of Edges")
    self.edgeList = tuple(sorted(edgeList, key=lambda x: x["nl"]))
    self.G = G
    self.dfsCode = computeDFSCode(G, edgeList)

    [source, target] = zip(*([e.tuple for e in edgeList]))
    vertexIndices = set(source + target)
    vertices = G.vs[vertexIndices]
    self.vertices = sorted(vertices, key=lambda x: x['nl'])

  def __eq__(self, other):
    return (isinstance(other, self.__class__) and
          self.__dict__ == other.__dict__)

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    [keys, values] = zip(*(sorted(self.__dict__.iteritems(), key=lambda x: x[0])))
    return hash(values)

  def getDFSCode(self):
    return self.dfsCode
   
  def getEdgeList(self):
    return self.edgeList

  def getVertices(self):
    return self.vertices

  def getChildPatterns(self):
    ''' returns child patterns with one edge growth '''
    #Get the list of neighbors that are not part of the pattern.
    #Sort the list in lexicographic order of vertex followed by edge labels.
    vertexIndices = set(map(lambda x: x.index, self.vertices))
    neighbors = [] # list of tuples of veritces and edges connecting the vertices to their corresponding vertices in pattern

    for v in self.vertices:
      for u in v.neighbors():
        if u.index not in vertexIndices:
          neighbors.append((u, getEdge(self.G, v, u)))
    
    neighbors = sorted(neighbors, key=lambda (v,e): (v["nl"], e["nl"]))
    childPatterns = []
    for e in zip(*neighbors)[1]:
      p = Pattern(self.G, list(self.edgeList) + [e])
      childPatterns.append(p)

    return childPatterns

if __name__ == '__main__':
  sys.exit(main())
