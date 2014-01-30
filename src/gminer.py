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
INSTANCE_GRAPH_NODES = {} # Stores the instance graph nodes for a given pattern.

def main():
  g = createTestGraph()
  #plotGraph(g, g.vs["l"], g.es["l"])
  preprocessGraph(g)
  
  p = Pattern(g, [g.es[0], g.es[1]])
  print "Initial pattern:"
  print p
  print "------------"

  ig = createInstanceGraph(g,p)
  print "Pattern instances:"
  for (k,v) in INSTANCE_GRAPH_NODES.iteritems():
    print "%d:"%v
    print k
    print k.__hash__()
  print "********"

  cp = p.getChildPatterns()
  print "Child patterns:"
  for c in cp:
    print c
  print "********"

  #newIg = growInstanceGraph(g, cp[0], p, ig)

  label = map(lambda (x,y): "%s_%d"%(x,y), zip(g.vs["nl"], g.vs.indices))
  plotGraph(g, label, g.es["nl"])
  plotGraph(ig, map(str,ig.vs.indices), map(str,ig.es.indices))
  #plotGraph(newIg, map(str,newIg.vs.indices), map(str,newIg.es.indices))

  '''
  #Create one edge graphs. S1 contains all the one edge graphs
  '''

def growInstanceGraph(g, pattern, parentPattern, parentIg):
  ''' Create the instance graph for pattern @pattern from its parent pattern @parentPattern whose
      instance graph is given by @parentIg '''

  childEdges = set([x.index for x in pattern.getEdgeList()]) 
  parentEdges = set([x.index for x in parentPattern.getEdgeList()])
  newEdgeIndex = childEdges.difference(parentEdges)

  dfsCode = str(pattern.getDFSCode())
  parentDfsCode = str(parentPattern.getDFSCode())
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
  for p in PATTERNS[parentDfsCode]:
    newPatternList = []
    pEdgeIndices = set([x.index for x in p.getEdgeList()])
    for gv in p.getVertices():
      if gv["nl"] != v0["nl"]:
        continue

      for gu in gv.neighbors():
        if gu["nl"] != v1["nl"]:
          continue

        e = getEdge(g, gv, gu)
        if e.index in pEdgeIndices:
          continue

        newPatternEdges = list(p.getEdgeList()) #TODO: Fix this.
        newPatternEdges.append(e)
        newPattern = Pattern(g, newPatternEdges)

        if newPattern in PATTERNS[dfsCode]:
          continue
        PATTERNS[dfsCode].add(newPattern)

        print newPattern
        print

        newPatternList.append(newPattern)
        newIg.add_vertex()
        vid = newIg.vs[-1:].indices[0]
        INSTANCE_GRAPH_NODES[newPattern] = vid
  
  #Create edges between vertices of newIg
  createInstanceGraphEdges(newIg, dfsCode) 
  return newIg

def createInstanceGraph(g, p):
  ''' Creates the instance of graph of pattern p. Where p is a pattern of two edges 
      as follow: v0-(e0)-v1, v1-(e1)-v2'''

  if len(p.getEdgeList()) != 2:
    raise Exception("More that 2 edges in pattern")

  p = p.getDFSCode()
  dfsCode = str(p)
  if dfsCode not in PATTERNS:
    PATTERNS[dfsCode] = set()
  ig = Graph() #Instance graph
  for e0 in g.es(nl_eq = p[1]):
    (v0, v1) = getVertices(g,e0)
    if v0["nl"] == p[0][1] and v1["nl"] == p[2][1]:
      for v2 in v1.neighbors():
        if v2.index == v0.index:
          continue
        e1 = getEdge(g, v1, v2)
        if e1.index == e0.index:
          continue

        if e1["nl"] == p[4] and v2["nl"] == p[5][1]:
          # Found a pattern
          newPattern = Pattern(g, [e0,e1])
          if newPattern in PATTERNS[dfsCode]:
            continue
          PATTERNS[dfsCode].add(newPattern)
          pid = addPattern(ig, newPattern)

  # Create edges between instance graph nodes.
  createInstanceGraphEdges(ig, dfsCode) 
  return ig
        
def createInstanceGraphEdges(ig, dfsCode):
  patternList = sorted(PATTERNS[dfsCode], key=lambda x: INSTANCE_GRAPH_NODES[x])
  print [INSTANCE_GRAPH_NODES[x] for x in patternList] #TODO: debug
  for i in range(0, (len(patternList) -1)):
    for j in range(i+1, len(patternList)):
      pi = patternList[i]
      pj = patternList[j]
      if pi.overlaps(pj):
        # Create edges between instance graph nodes corresponding to pi and pj
        ig.add_edge(INSTANCE_GRAPH_NODES[pi], INSTANCE_GRAPH_NODES[pj])
        

def addPattern(g, pattern):
  ''' Adds the pattern identified by the two edges to the instance graph g.
      In case edges are shared between patterns then connections are created.'''
  g.add_vertex()
  pid = len(g.vs) - 1 #pid represent the instance graph node that was added
  if pattern in INSTANCE_GRAPH_NODES:
    raise Exception("Duplicate pattern")
  INSTANCE_GRAPH_NODES[pattern] = pid


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
  # Create edge labels
  #edge_labels = ["a", "a", "a", "a", "b", "b", "b", "c", "d"]
  edge_labels = ["a"]
  g.vs["l"] = vertex_lables
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
  dfsCodeList = []
  for v in [x for x in vertices if x['nl']==vertices[0]['nl']]:
      dfsCode = []
      v["visited"] = 0
      time = 0
      dfsCode = dfsVisit(G, v, dfsCode, vertexSet, edgeSet, time)
      dfsCodeList.append(DfsCode(dfsCode))

      del G.vs["visited"]
      del G.es["forward"]
  dfsCodeList = sorted(dfsCodeList, key=lambda x: str(x))
  return dfsCodeList[0] # Returns the minimun DFSCode

def dfsVisit(G, vertex, dfsCode, vertexSet, edgeSet, time):
  neighbors = sorted(vertex.neighbors(), key=lambda x: (x.attributes().get("visited", sys.maxsize),
                              getEdge(G, vertex, x)["nl"],
                              x["nl"])) # First get backward edges, and the sort 
                              #forward edges by edge label followed by vertex label
  for u in neighbors:
    if u.index not in vertexSet:
      continue
    e = getEdge(G, vertex, u)
    if e.index not in edgeSet:
      continue
    
    if "forward" in e.attributes() and e["forward"] == True:
      continue

    if "visited" in u.attributes() and u["visited"]:
      #This is a backward edge
      if u["visited"] > vertex["visited"]:
        raise Exception("Backward edge (u,v): (u)%d should be less than (v)%d"%(vertex["visited"], u["visited"])
      dfsCode = visitEdge(G, (vertex, u), dfsCode) 
    else:
      #This is a forward edge
      time += 1
      u["visited"] = time
      dfsCode = visitEdge(G, (vertex, u), dfsCode) 
      dfsCode = dfsVisit(G, u, dfsCode, vertexSet, edgeSet, time)
    
  return dfsCode
      
def visitEdge(G, e, dfsCode):
  ''' The dfscode is of the form: 
  [(<visit time>,<vertex label>),<edge label>,(<visit time>, <vertex label>)]
  '''
  e["forward"] = True
  v = [G.vs[e.source], G.vs[e.target]]
  [v0, v1] = sorted(v, key=lambda x: x["visited"])
  l = [(v0["visited"], v0["nl"]), e["nl"], (v1["visited"], v1["nl"])]
  dfsCode += l
  return dfsCode

class DfsCode:
  ''' A class for DFSCode which encapsulates the order in which vertices and edges 
      are visited as a list. '''
  def __init__(self, dfsOrder):
    if type(dfsOrder) != list:
      raise Exception("The dfs order should be a list")
    self.dfsCode = tuple(dfsOrder)
    s = []
    for x in self.dfsCode:
      if type(x) == tuple:
        s.append("%d_%s"%(x[0], x[1]))
      else:
        s.append(str(x))
    self.dfsCodeString = ".".join(s)

        
  def __eq__(self, other):
    return ((self.dfsCode == other.dfsCode) and
      (self.dfsCodeString == other.dfsCodeString))

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash((self.dfsCode, self.dfsCodeString))

  def __str__(self):
    return self.dfsCodeString

  def __getitem__(self, i):
    return self.dfsCode[i]

class Pattern:
  ''' A list of edges repsenting a pattern. The list of edges are sorted according to the DFS order. The DFS code of the
  pattern is also persisted. '''

  def __init__(self, G, edgeList):
    if type(edgeList) != list:
      raise Exception("argument must be a list of Edges")
    self.edgeList = tuple(sorted(edgeList, key=lambda x: (x["nl"], x.index)))
    self.G = G
    self.dfsCode = computeDFSCode(G, edgeList)

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
    val += "DFS Code: %s\n"%(str(self.dfsCode))
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
    edgeIndices = set(map(lambda x: x.index, self.edgeList))
    rightmostVertex = self.G.vs[sorted(self.dfsOrder[-1], key=lambda x: x[0])[1][1]]
    rightmostPath = [rightmostVertex] 
    ''' The DFS Order is as follows:
      [((t(v0), v0), (t(v1), v1)),
       ((t(v1), v1), (t(v2), v2)),
        ...]
    '''
    currentVertex = rightmostVertex
    for e in self.dfsOrder[::-1]:
      if e[0][0] > e[1][1]:
        #This is a backward edge. Ignore.
        continue
      if e[1][1] == currentVertex:
        self.G.vs[rightmostPath.append(e[0][1])]
        currentVertex = e[0][1]
      

    #Check for backward edges from the rightmost vertex.
    for v in rightmostVertex.neighbors():
      e = getEdge(self.G, rightmostVertex, v)
      if e.index in edgeIndices:
        continue
      

    return childPatterns

  def overlaps(self, other):
    ''' Returns true if the edges of the pattern overlaps that of the @other '''
    edgeList = set([e.index for e in self.edgeList])
    otherEdgeList = set([e.index for e in other.getEdgeList()])
    return not edgeList.isdisjoint(otherEdgeList)


if __name__ == '__main__':
  sys.exit(main())

#TODO: When computing single edge graphs. For each edge have two patterns where the direction of traversal of the pattern is
# switched for the two patterns.
