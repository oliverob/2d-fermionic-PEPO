using Graphs
using MetaGraphsNext
using GraphPlot, Compose
using QuantumAlgebra

function convert_to_PEPO(G)
    new_graph = MetaGraph(DiGraph(), label_type=Int, vertex_data_type=Tuple{Symbol, Int}, edge_data_type=Int)
    for vertex in vertices(G)
        new_graph[vertex] = (:v, 1)
    end
    current_edges = edges(G)
    for edge in current_edges
        u, v = src(edge), dst(edge)
        new_graph[nv(new_graph)+1] = (:e,0)
        new_graph[u, nv(new_graph)] = new_graph[u][2]
        new_graph[u] = (:v, new_graph[u][2]+1)
        new_graph[nv(new_graph), v] =  new_graph[v][2]
        new_graph[v] = (:v, new_graph[v][2]+1)
    end
    return new_graph
end

n = 10
k = 2
G = grid([4,4])

PEPO = convert_to_PEPO(G)

if (!is_connected(G))
    throw("Not connected")
end

t = gplot(PEPO, edgelabel=map((x)-> x[2],sort(collect(PEPO.edge_data), by=x->x[1])), edgelabelc="white", EDGELABELSIZE=8, nodelabel=map((x)-> x[2][2][1],sort(collect(PEPO.vertex_properties), by=x->x[1])))

t = gplot(PEPO, edgelabel=map((x)-> x[2],sort(collect(PEPO.edge_data), by=x->x[1])), edgelabelc="white", EDGELABELSIZE=8, nodelabel=map((x)-> x[2][1],sort(collect(PEPO.vertex_properties), by=x->x[1])))
gplot(G)

start = 11
target = 7
shortest_path = a_star(G, start, target)
edges(PEPO)

function has_order_higher_than_path(source,path, leg)
    path_edge = (source, path)
    if !has_edge(PEPO, path_edge...)
        path_edge = (path, source)
    end
    leg_edge = (source, leg)
    if !has_edge(PEPO,leg_edge...)
        leg_edge = (leg, source)
    end
    return PEPO[leg_edge...] > PEPO[path_edge...]
end
operators = σz()^2
for edge in shortest_path
    source = src(edge) # Vertex tensor
    destination = dst(edge) # Vertex tensor
    src_neighbours = all_neighbors(PEPO, src(edge))
    dest_neighbours = all_neighbors(PEPO, dst(edge))
    path = intersect(src_neighbours, dest_neighbours)[1] # GHZ tensor
    z_edges_src = filter((x)-> has_order_higher_than_path(source,path, x),src_neighbours)
    z_edges_dst = filter((x)-> has_order_higher_than_path(destination,path, x),dest_neighbours)
    operators *= has_edge(PEPO,source, path) ? 1 : -1
    operators *= prod(map((x)-> σz(x),z_edges_src))
    operators *= prod(map((x)-> σz(x),z_edges_dst))
    operators *= σy(path)
end
println(normal_form(operators))

 