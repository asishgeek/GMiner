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

PATTERNS = {} #Stores the edgeSets for each pattern specified by the DFS code.

def main():
  g = createTestGraph()
  #plotGraph(g, g.vs["l"], g.es["l"])
  preprocessGraph(g)
  
  p = Pattern(g, [g.es[0], g.es[1]])
  print "Initial pattern:"
  print p
  print "------------"

  ig = createInstanceGraph(g,p)
  cp = p.getChildPatterns()
  print "Child patterns:"
  for c in cp:
    print c
  print "-------------"

  newIg = growInstanceGraph(g, cp[0], p, ig)

  label = map(lambda (x,y): "%s_%s"%(x,y), zip(g.vs["nl"], g.vs["indices"]))
  plotGraph(g, label, g.es["nl"])
  #plotGraph(ig, map(str,ig.vs.indices), map(str,ig.es.indices))

  '''
  #Create one edge graphs. S1 contains all the one edge graphs
  '''

def growInstanceGraph(g, pattern, parentPattern, parentIg):
  ''' Create the instance graph for pattern @pattern from its parent pattern @parentPattern whose
      instance graph is given by @parentIg '''

  childEdges = set([x.index for x in pattern.getEdgeList()]) 
  parentEdges = set([x.index for x in parentPattern.getEdgeList()])
  newEdgeIndex = childEdges.difference(parentEdges)

  dfsCode = ".".join(pattern.getDFSCode())
  print pattern.getDFSCode()

  if dfsCode not in PATTERNS:
    PATTERNS[dfsCode] = set()

  if not newEdgeIndex:
    return None
  if len(newEdgeIndex) != 1:
    raise Exception("Cannot grow instance graph beucase has child pattern has %d edges more than the parent pattern"%len(newEdgeIndex))
  
  newEdge = g.es[newEdgeIndex.pop()]
  [v0, v1] = getVertices(g, newEdge) #V0 and V1 are the vertices of the new edge that is present in the child pattern

  newIg = Graph()
  for v in parentIg.vs:
    cliqueSize = 0
    newPatternList = []
    for gv in v["G"]:
      if gv["nl"] != v0["nl"]:
        continue

      for gu in gv.neighbors():
        if gu["nl"] != v1["nl"]:
          continue

        e = getEdge(g, gv, gu)
        if e.index in parentEdges:
          continue

        newPatternEdges = list(parentPattern.getEdgeList()) #TODO: Fix this.
        newPatternEdges.append(e)
        newPattern = Pattern(g, newPatternEdges)

        if newPattern in PATTERNS[dfsCode]:
          continue
        PATTERNS[dfsCode].add(newPattern)

        print newPattern
        print

        cliqueSize += 1
        newPatternList.append(newPattern)
        
    nv = addClique(newIg, cliqueSize, newPatternList)
    if nv:
      v["child"] = nv
  
  #Create edges between vertices of newIg
  for eig in parentIg.es:
    [v0, v1] = [v for v in parentIg.vs[eig.tuple]]
    if "chlid" in v0.attributes() and "chlid" in v1.attributes():
      cv0 = v0["child"] #cv0 is a node in newIg
      cv1 = v1["child"] #cv1 is a node in newIg
      newIg.add_edge(cv0, cv1)

  if "child" in parentIg.vs.attributes():
    del parentIg.vs["child"]

  return newIg

def addClique(g, cliqueSize, newPatternList):
  if cliqueSize <= 0:
    return None
  
  g.add_vertex()
  vid = g.vs.indices[-1] # Id of the last added vertex

  # Make links back to the original graph
  vertices = [v.index for v in newPatternList[0].getVertices()]
  for v in vertices:
    g.vs[vid]["G"] = v

  for i in range(1, cliqueSize):
    g.add_vertex()
    # Make links back to the original graph
    vertices = [v.index for v in newPatternList[i].getVertices()]
    for v in vertices:
      g.vs[vid+i]["G"] = v
    
    for j in range(i-1,-1,-1):
      g.add_edge(vid + j, vid + i)

def createInstanceGraph(g, p):
  ''' Creates the instance of graph of pattern p. Where p is a pattern of two edges 
      as follow: v0-(e0)-v1-(e1)-v2'''

  if len(p.getEdgeList()) != 2:
    raise Exception("More that 2 edges in pattern")

  p = p.getDFSCode()
  dfsCode = ".".join(p)
  if dfsCode not in PATTERNS:
    PATTERNS[dfsCode] = set()
  ig = Graph() #Instance graph
  for e0 in g.es(nl_eq = p[1]):
    (v0, v1) = getVertices(g,e0)
    if v0["nl"] == p[0] and v1["nl"] == p[2]:
      for v2 in v1.neighbors():
        if v2.index == v0.index:
          continue
        e1 = getEdge(g, v1, v2)
        if e1.index == e0.index:
          continue

        if e1["nl"] == p[3] and v2["nl"] == p[4]:
          # Found a pattern
          newPattern = Pattern(g, [e0,e1])
          if newPattern in PATTERNS[dfsCode]:
            continue
          PATTERNS[dfsCode].add(newPattern)
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
  vertices = sorted(vertices, key=lambda(x): (g.vs[x]["nl"], x)) #Sort by lexicographic orider of labels
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
  edgeSet = set([e.index for e in edgeList])
  dfsCode = []
  for v in vertices:
    if "visited" not in v.attributes():
      dfsCode = visit(v, dfsCode)
      dfsCode = dfsVisit(G, v, dfsCode, vertexSet, edgeSet)

  del G.vs["visited"]
  del G.es["forward"]
  return dfsCode

def dfsVisit(G, vertex, dfsCode, vertexSet, edgeSet):
  neighbors = sorted(vertex.neighbors(), key=lambda x: (x["nl"], getEdge(G, vertex, x)["nl"])) #Sort by vertex label followed by edge label
  for u in neighbors:
    if u.index not in vertexSet:
      continue
    e = getEdge(G, vertex, u)
    if e.index not in edgeSet:
      continue
    if "visited" in u.attributes() and u["visited"] == True:
      continue

    dfsCode = visit(e, dfsCode) 
    dfsCode = visit(u, dfsCode)
    dfsCode = dfsVisit(G, u, dfsCode, vertexSet, edgeSet)

  #Get backward edges and add it in the DFS code
  for u in neighbors:
    if u.index not in vertexSet:
      continue
    e = getEdge(G, vertex, u)
    if e.index not in edgeSet:
      continue
    if "forward" in e.attributes() and e["forward"] == True:
      continue
    dfsCode = visit(e, dfsCode)
    dfsCode.append(u["nl"])
    
  return dfsCode
      
def visit(item, dfsCode):
  #print "visiting item: %s"%str(item)
  if type(item) == igraph.Edge:
    item["forward"] = True
  elif type(item) == igraph.Vertex:
    item["visited"] = True
  else:
    raise Exception("Invalid type of item should be either a vertex or an edge")
  dfsCode.append(item["nl"])
  return dfsCode

class Pattern:
  ''' A list of edges repsenting a pattern. The list of edges are sorted according to the DFS order. The DFS code of the
  pattern is also persisted. '''

  def __init__(self, G, edgeList):
    if type(edgeList) != list:
      raise Exception("argument must be a list of Edges")
    self.edgeList = tuple(sorted(edgeList, key=lambda x: (x["nl"], x.index)))
    self.G = G
    self.dfsCode = tuple(computeDFSCode(G, edgeList))

    [source, target] = zip(*([e.tuple for e in edgeList]))
    vertexIndices = set(source + target)
    vertices = G.vs[vertexIndices]
    self.vertices = tuple(sorted(vertices, key=lambda x: (x['nl'], x.index)))

  def __key(self):
    return (self.G.__hash__(),
            tuple([e.index for e in self.edgeList]),
            self.dfsCode,
            tuple([v.index for v in self.vertices]))

  def __eq__(self, other):
    return (isinstance(other, self.__class__) and
          self.__key() == other.__key())

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(self.__key())

  def __str__(self):
    val = "Graph: %d\n"%(self.G.__hash__())
    val += "Edge List: %s\n"%(", ".join([str(e.index) for e in self.edgeList]))
    val +="Vertices: %s\n"%(", ".join([str(v.index) for v in self.vertices]))
    val += "DFS Code: %s\n"%(".".join(self.dfsCode))
    return val

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
