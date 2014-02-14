#!/usr/bin/python
import sys
import operator
import igraph
from igraph import Graph
from igraph.clustering import VertexClustering
from collections import OrderedDict

''' GMiner algorithm. http://dl.acm.org/citation.cfm?id=1598339
Original labels of vertices and edges are identified by the attribute "l".
Frequency based labels are identified by the attribute "nl".
'''

MIN_FREQUENCY = 1

PATTERN_INSTANCES = OrderedDict() # Stores the pattern instances for a pattern.


class Pattern:
  ''' A Pattern is a sequence of edges (DFS Code) identified by edge and vertex labels.
      Each item in the sequence is an edge (u,v) represented as a 5 tuple:
        (t(u), t(v), l(u), l((u,v)), l(v)) where t represents the time of traversal and l is the label.
      Two patterns are the same if they encode the same set of edges even if the sequence of edge
      traversal is different. The sequence gives the DFS Code of the pattern.
  '''

  def __init__(self, edgeSequence):
    self.edgeSequence = list(edgeSequence)
    #Compute the degree distribution so to distinguish between same patterns having different
    # DFS Codes.
    degreeDistributionMap = {}
    for (tu, tv, lu, le, lv) in edgeSequence:
      count = degreeDistributionMap.get((tu, lu), 0) 
      degreeDistributionMap[(tu, lu)] = count + 1
      count = degreeDistributionMap.get((tv, lv), 0) 
      degreeDistributionMap[(tv, lv)] = count + 1
    
    degreeDist = []
    for (k, v) in degreeDistributionMap.iteritems():
      degreeDist.append((k[1], v))

    self.degreeDistribution = tuple(sorted(degreeDist))
    self.key = (tuple(sorted(map(lambda x: tuple(sorted(x[2:])), self.edgeSequence))) , self.degreeDistribution)
    

  def __key(self):
    ''' Returns the key that is used for checking equality. Only the vertex and edge labels are 
        considered for checking eqality. Meaning that two patterns having the same set of edges
        are considered equal even if the order of traversal is different '''
    return self.key

  def __eq__(self, other):
    return (isinstance(other, self.__class__) and
            self.__key() == other.__key())

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(self.__key())

  def __str__(self):
    return str(self.edgeSequence)

  def dfsCode(self):
    return tuple(self.edgeSequence)

class PatternInstance:
  ''' Represents a node in the instance graph for a specific pattern.
      The node encapsulates the set of edges that have the same pattern (DFS Code).
  '''
  
  def __init__(self, G, dfsTree):
    ''' dfsTree is a sequence of edges represented as a 4 tuple (t(u), t(v), u, v)
        where t(u) represents the time at which vertex u was visited and u and v represent 
        actual vertex indices in the graph G.
    '''
    self.dfsTree = tuple(dfsTree)
    edgeList = []
    vertexList = []
    for (tu, tv, u, v) in dfsTree:
      if type(u) != int or type(v) != int:
        raise Exception("Integer vertex index expected")
      edgeList.append(G.get_eid(u, v))  
      vertexList.append(u)
      vertexList.append(v)
    
    self.edgeSet = set(edgeList)
    self.vertexSet = set(vertexList)
    
  def __key(self):
    return self.edgeSet

  def __eq__(self, other):
    return (isinstance(other, self.__class__) and
            self.__key() == other.__key())

  def __ne__(self, other):
    return not self.__eq__(other)

  def __hash__(self):
    return hash(tuple(self.__key()))
    
  def __str__(self):
    rep = "Edge set: %s\n"%(str(self.edgeSet))
    rep += "DFS Tree: %s\n"%(str(self.dfsTree))
    return rep

  def overlaps(self, other):
    ''' Returns true if the pattern instance overlaps the other pattern instance. False otherwise. '''
    return (len(self.edgeSet.intersection(other.edgeSet)) != 0)

def getChildPatterns(G, p):
  ''' Returns child patterns of p formed by one edge growth. Returns only patterns that have minimum DFS Code '''
  childPatterns = OrderedDict()
  if p not in PATTERN_INSTANCES:
    return None

  for instance in PATTERN_INSTANCES[p]:
    rpath = getRightmostPath(instance) # Returns a sequence of (t(v), v) tuples from the root to the rightmost vertex. 
    instanceEdges = instance.edgeSet
    instanceVertices = zip(*(map(lambda x: (G.es[x].source, G.es[x].target), instanceEdges)))
    instanceVertices = set(instanceVertices[0] + instanceVertices[1])

    #First check backward edges from the rightmost vertex.
    (trv, rv) = rpath[-1] #Rightmost vertex
    for (tv, v) in rpath[0:-1]:
      if G[rv,v] == 1:
        # An edge exists between the rightmost vertex and a vertex in the rightmost path
        eid = G.get_eid(v,rv)
        if eid in instanceEdges:
          continue
        
        cp = Pattern(p.edgeSequence + [(trv, tv, G.vs[rv]["nl"], G.es[eid]["nl"], G.vs[v]["nl"])])
        if cp in PATTERN_INSTANCES:
          #This is not the minimum DFS Code. 
          continue

        if cp.dfsCode() in childPatterns:
          continue
        childPatterns[cp.dfsCode()] = cp
     
    # Next check for forward edges from vertices in the rightmost path
    for (tv, v) in rpath[::-1]:
      for u in sorted(map(lambda x: x.index, G.vs[v].neighbors()), 
                      key=lambda x: (G.es[G.get_eid(v,x)]["nl"], G.vs[x]["nl"])):

        if u in instanceVertices:
          continue

        
        eid = G.get_eid(v,u)
        cp = Pattern(p.edgeSequence + [(tv, trv+1,  G.vs[v]["nl"], G.es[eid]["nl"], G.vs[u]["nl"])])
        if cp in PATTERN_INSTANCES:
          #This is not the minimum DFS Code. 
          continue

        if cp.dfsCode() in childPatterns:
          continue
        childPatterns[cp.dfsCode()] = cp

  return childPatterns.values()
        

def getRightmostPath(patternInstance):
  rpath = []
  rv = None #Rightmost vertex
  trv = -1 #time of rightmost vertex
  index = len(patternInstance.dfsTree) - 1
  for (tu, tv, u, v) in patternInstance.dfsTree[::-1]:
    if tv > tu:
      rv = v
      trv = tv
      break
    index -= 1

  rpath.append((trv, rv))
  cv = rv #Current vertex
  for (tu, tv, u, v) in patternInstance.dfsTree[index::-1]:
    if tv > tu and cv == v:
      cv = u
      rpath.append((tu, cv))

  rpath = rpath[::-1] #Reverse the list
  return rpath

def createTwoEdgeInstances(g, pattern):
  ''' Create instance graph for the two edge @pattern where @pattern is a minimum pattern.'''
  patternLabels = pattern.edgeSequence
  if len(patternLabels) != 2:
    raise Exception("Pattern has more than two edges.")

  for e in g.es:
    (u, v) = sorted([g.vs[e.source], g.vs[e.target]], key=lambda x: x["nl"])

    p0 = (0, 1, u["nl"], e["nl"], v["nl"])
    if p0 != patternLabels[0]:
      continue

    edges = [(u,v)]
    if u["nl"] == v["nl"]:
      edges.append((v,u))

    for (u,v) in edges:
      for x in v.neighbors():
        if x.index == u.index:
          continue
        e1 = g.es[g.get_eid(v.index,x.index)]
        p1 = (1, 2, v["nl"], e1["nl"], x["nl"])
        if p1 != patternLabels[1]:
          continue

        #We found a pattern
        newPattern = Pattern([p0, p1])
        instanceDfsTree = [(0, 1, u.index, v.index), (1, 2, v.index, x.index)]
        newPatternInstance = PatternInstance(g, instanceDfsTree)

        instanceNodeList = PATTERN_INSTANCES.get(pattern, set([]))
        instanceNodeList.add(newPatternInstance)
        PATTERN_INSTANCES[pattern] = instanceNodeList

      for x in u.neighbors():
        if x.index == v.index:
          continue
        e1 = g.es[g.get_eid(u.index,x.index)]
        p1 = (0, 2, u["nl"], e1["nl"], x["nl"])
        if p1 != patternLabels[1]:
          continue

        #We found a pattern
        newPattern = Pattern([p0, p1])
        instanceDfsTree = [(0, 1, u.index, v.index), (0, 2, u.index, x.index)]
        newPatternInstance = PatternInstance(g, instanceDfsTree)
        
        instanceNodeList = PATTERN_INSTANCES.get(pattern, set([]))
        instanceNodeList.add(newPatternInstance)
        PATTERN_INSTANCES[pattern] = instanceNodeList
      
  
def getOneEdgeSubgraph(g):
  edges = []
  for e in g.es:
    (u, v) = sorted([g.vs[e.source], g.vs[e.target]], key=lambda x: x["nl"])
    edges.append((u, v))

  edges = sorted(edges, key=lambda (x, y): (x["nl"], y["nl"]))
  return edges

def getPatternInstances(g, pattern, parentPattern):
  if pattern in PATTERN_INSTANCES:
    raise Exception("Pattern Already present: %s"%str(pattern))

  childInstanceParentMap = {} #Stores the parent instance (value) for a child instance (key)

  childEdge = (set(pattern.edgeSequence)).difference(set(parentPattern.edgeSequence))
  if len(childEdge) > 1:
    raise Exception("Child pattern has more than one extra edge from parent pattern")

  childEdge = childEdge.pop()
  (tu, tv, lu, le, lv) = childEdge

  for instance in PATTERN_INSTANCES[parentPattern]:
    rpath = getRightmostPath(instance)
    rmostVertex = g.vs[rpath[-1][1]]
    rmostTime = rpath[-1][0]
    # Check for backward edges.
    if tu > tv:
      #This is a backward edge. We only need to search from the rightmost vertex.
      if rmostTime == tu and rmostVertex["nl"] == lu:
        for (tx, x) in rpath[0:-1]:
          if g[rmostVertex.index, x] == 0: #Not a valid edge
            continue
          eindex = g.get_eid(rmostVertex.index, x)
          if eindex in instance.edgeSet:
            continue

          if tx == tv and lv == g.vs[x]["nl"] and le == g.es[eindex]["nl"]:
            newInstance = PatternInstance(g, list(instance.dfsTree) + [(tu, tv, rmostVertex.index, x)])
            childInstanceParentMap[newInstance] = instance
      continue
      
    #Otherwise Check for forward edges
    #FInd the appropriate vertex in the parent instance graph from which this vertex can be extended.
    for (tx, x) in rpath:
      if tx == tu and lu == g.vs[x]["nl"]:
        for y in g.vs[x].neighbors():
          eindex = g.get_eid(y.index, x)
          if (eindex in instance.edgeSet) or (y.index in instance.vertexSet):
            continue

          if le == g.es[eindex]["nl"] and lv == y["nl"]:
            newInstance = PatternInstance(g, list(instance.dfsTree) + [(tu, tv, x, y.index)])
            childInstanceParentMap[newInstance] = instance
  return childInstanceParentMap

def createInstanceGraphForPattern(pattern):
  instanceGraph = Graph()
  if pattern not in PATTERN_INSTANCES:
    raise Exception('Pattern %s not found'%str(pattern))

  patternInstances = list(PATTERN_INSTANCES[pattern])
  for instance in patternInstances:
    instanceGraph.add_vertex()
    vertex = instanceGraph.vs[len(instanceGraph.vs) - 1]
    vertex["instance"] = instance
  
  for i in range(0, len(patternInstances) - 1):
    for j in range(i+1, len(patternInstances)):
      instance1 = patternInstances[i]
      instance2 = patternInstances[j]
      if instance1.overlaps(instance2):
        instanceGraph.add_edge(i,j)

  return instanceGraph

def createTwoEdgeGraphs(g):
  for (u, v) in getOneEdgeSubgraph(g):
    for x in v.neighbors():
      if x.index == u.index:
        continue
      p = Pattern([(0, 1, u["nl"], g.es[g.get_eid(u.index, v.index)]["nl"], v["nl"]),
                   (1, 2, v["nl"], g.es[g.get_eid(v.index, x.index)]["nl"], x["nl"])])
      if p in PATTERN_INSTANCES:
        continue

      PATTERN_INSTANCES[p] = set([])
      createTwoEdgeInstances(g, p)
      for cp in getChildPatterns(g, p):
        if cp in PATTERN_INSTANCES:
          continue
        childInstanceParentMap = getPatternInstances(g, cp, p)
        allInstances = set(childInstanceParentMap.keys())
        PATTERN_INSTANCES[cp] =  allInstances

    for x in u.neighbors():
      if x.index == v.index:
        continue
      p = Pattern([(0, 1, u["nl"], g.es[g.get_eid(u.index, v.index)]["nl"], v["nl"]),
                   (0, 2, u["nl"], g.es[g.get_eid(u.index, x.index)]["nl"], x["nl"])])
      if p in PATTERN_INSTANCES:
        continue

      PATTERN_INSTANCES[p] = set([])
      createTwoEdgeInstances(g, p)
      for cp in sorted(getChildPatterns(g, p), key=lambda x: x.dfsCode()):
        if cp in PATTERN_INSTANCES:
          continue
        childInstanceParentMap = getPatternInstances(g, cp, p)
        allInstances = set(childInstanceParentMap.keys())
        PATTERN_INSTANCES[cp] =  allInstances

def isNeighboringCluster(cluster1, cluster2, g):
  #Check if clusters 1 and 2 are neighbors. i.e. they have vertices that share an edge
  for u in cluster1:
    for v in cluster2:
      if g[u,v] == 1:
        return True
  return False
  
def mergePartitions(clustering, g):
  modularity = clustering.modularity
  newMembership = clustering.membership
  clusterIndices = list(set(clustering.membership))
  for i in range(0,len(clusterIndices) - 1):
    for j in range(i+1, len(clusterIndices)):
      #Check if the ith and jth clusters are neighbors. i.e. they have vertices that share an edge
      cxi = clusterIndices[i]
      cxj = clusterIndices[j]
      if isNeighboringCluster(clustering[cxi], clustering[cxj], g):
        #Merge clusters and see if the modularity value increases
        m = map(lambda x: cxi if x==cxj else x, newMembership) #Replaces cluster index j with i
        newClustering = igraph.clustering.VertexClustering(g, membership=m)
        if newClustering.modularity <= modularity:
          continue
        modularity = newClustering.modularity
        newMembership = m
  if id(clustering.membership) == id(newMembership):
    #The modularity didn't increase. So return the original clustering.
    return clustering

  # Renumber cluster ids so that cluster ids are continuous from 0.
  clusterIdMap = {}
  currentId = -1
  renumberedMembership = []
  for cid in newMembership:
    if cid not in clusterIdMap:
      currentId += 1
      clusterIdMap[cid] = currentId

    renumberedMembership.append(clusterIdMap[cid])

  return VertexClustering(g, membership=renumberedMembership)
 
def subgraphMining(g, pattern, ig, clustering, frequentPatterns):
  clustering = mergePartitions(clustering, ig)
  gMeasure = len(clustering)
  if gMeasure < 2: #TODO: Remove hadcoded threshold
    return

  frequentPatterns.add(pattern)

  #Get child patterns
  for cp in getChildPatterns(g, pattern):
    if cp in PATTERN_INSTANCES:
      continue
    childInstanceParentMap = getPatternInstances(g, cp, pattern)
    allInstances = set(childInstanceParentMap.keys())
    PATTERN_INSTANCES[cp] = allInstances

    #Create instance graph of child pattern
    childIg = createInstanceGraphForPattern(cp)

    #Assign child instances to clusters based on parent clusters
    clusterId = 0
    membership = []
    parentClusterIds = {}
    for cluster in clustering: #This is the clustering of the parent instance graph.
      for vertexIndex in cluster:
        parentInstance = ig.vs[vertexIndex]["instance"]
        parentClusterIds[parentInstance] = clusterId;
      clusterId += 1

    for childVertex in childIg.vs:
      childInstance = childVertex["instance"] 
      membership.append(parentClusterIds[childInstanceParentMap[childInstance]])

    childIgClustering = VertexClustering(childIg, membership=membership)

    subgraphMining(g, cp, childIg, childIgClustering, frequentPatterns)
  return

def printInstances(g, p):
  '''Print the instances of pattern @p. Used for debugging '''
  print "Pattern = %s:"%str(p)
  for i in PATTERN_INSTANCES[p]:
    print "%s"%str(i)

def gminer(g, t, s):
  ''' The main algorithm. @g is the graph, @t is the threshold, @s is the frequent subgraph set. '''
  frequentPatterns = set([])
  createTwoEdgeGraphs(g)
  twoEdgePatterns = PATTERN_INSTANCES.keys()
  for pattern in twoEdgePatterns:
    ig = createInstanceGraphForPattern(pattern)
    clustering = ig.components()
    subgraphMining(g, pattern, ig, clustering, frequentPatterns)

  return frequentPatterns

########## Preprocessing ################
def preprocessGraph(g):
  ''' Delete infrequent vertices and edges. Create new labels based on descending frequency. '''
  #Delete infrequent vertices/edges. Returns vertex and edge frequencies
  (vertexCount, edgeCount) = deleteInfrequentVerticesAndEdges(g)
  
  #Sort the vertex and edge counts in descending order
  vertexCount = sorted(vertexCount, key=operator.itemgetter(1), reverse=True)
  edgeCount = sorted(edgeCount, key=operator.itemgetter(1), reverse=True)
  
  #Re-label graph according to descending order of vertex and edge frequencies 
  return relabelGraph(g, vertexCount, edgeCount)

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
  vertexLabelMap = {}
  count = 0
  for (label, freq) in vertexCount:
    newLabel = "V%d"%count
    vertexLabelMap[newLabel] = label
    for v in g.vs(l_eq=label):
      v["nl"] = newLabel #nl stands for new label
    count += 1

  edgeLabelMap = {}
  count = 0
  for (label, freq) in edgeCount:
    newLabel = "E%d"%count
    edgeLabelMap[newLabel] = label
    for e in g.es(l_eq=label):
      e["nl"] = newLabel #nl stands for new label
    count += 1
  return (vertexLabelMap, edgeLabelMap)

#########Test Harness ##################

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

def relabelPattern(p, vertexLabelMap, edgeLabelMap):
  newEdgeSequqence = map(lambda (tu, tv, lu, le, lv): (tu, tv, vertexLabelMap[lu], edgeLabelMap[le], vertexLabelMap[lv]),
    p.edgeSequence)
  return Pattern(newEdgeSequqence)
  

def main():
  g = createTestGraph()

  (vertexLabelMap, edgeLabelMap) = preprocessGraph(g)

  frequentPatterns = gminer(g, 1, None)
  print "Frequent Patterns:"
  for p in frequentPatterns:
    np = relabelPattern(p, vertexLabelMap, edgeLabelMap)
    print "Pattern = %s, count = %d"%(np, len(PATTERN_INSTANCES[p]))
  
  
if __name__ == "__main__":
  sys.exit(main())
#########################################
