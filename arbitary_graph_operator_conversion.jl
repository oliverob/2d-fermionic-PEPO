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

function fermionic_XX(PEPO,start, target)
    shortest_path = a_star(G, start, target)

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
    return normal_form(operators)
end

function fermionic_Z(PEPO, target)
    operator = σz()^2
    for ghz_tensor in all_neighbors(PEPO, target)
        operator *= σz(ghz_tensor)
    end
    return normal_form(operator)
end

l = 4
G = smallgraph(:petersen)
G = Graph(kruskal_mst(G))

PEPO = convert_to_PEPO(G)

if (!is_connected(G))
    throw("Not connected")
end

t = gplot(PEPO, edgelabel=map((x)-> x[2],sort(collect(PEPO.edge_data), by=x->x[1])), edgelabelc="white", EDGELABELSIZE=8, nodelabel=map((x)-> x[2][2][1],sort(collect(PEPO.vertex_properties), by=x->x[1])))

t = gplot(PEPO, edgelabel=map((x)-> x[2],sort(collect(PEPO.edge_data), by=x->x[1])), edgelabelc="white", EDGELABELSIZE=8, nodelabel=map((x)-> x[2][1],sort(collect(PEPO.vertex_properties), by=x->x[1])))

gplot(G)

fermionic_XX(PEPO,1, 2)

total_pauli_weight = 0
total_hopping_terms = 0
# Testing even fermionic algebra
for i in 1:nv(G)
    for j in 1:nv(G)
        if i == j
            continue
        end
        current_hopping_term = fermionic_XX(PEPO,i, j)
        total_pauli_weight += length(collect(keys(current_hopping_term.terms))[1].bares.v)
        total_hopping_terms += 1
        for k in i:nv(G)
            for l in j:nv(G)
                other_hopping_term = fermionic_XX(PEPO, k,l)
                if (i == k ⊻ j == l)
                    if normal_form(comm(current_hopping_term, other_hopping_term)) == 0*one(σx())
                        throw("Error two touching terms commute")
                    end
                elseif i == l && j == k && i != j
                    if normal_form(current_hopping_term) != -normal_form(other_hopping_term)
                        throw("Error two opposite edges are not opposite signed $current_hopping_term, $other_hopping_term")
                    end
                elseif i != k && j != l && i != l && j != k && i != j && l != k
                    if normal_form(comm(current_hopping_term, other_hopping_term)) != 0*one(σx())
                        throw("Error two non-touching terms anticommute $i, $j, $k, $l")
                    end
                end
            end
            parity_term = fermionic_Z(PEPO, k)
            if (i == k ⊻ j == k)
                if normal_form(comm(current_hopping_term, parity_term)) == 0*one(σx())
                    throw("Error two touching terms commute")
                end
            elseif i != k && j != k
                if normal_form(comm(current_hopping_term, parity_term)) != 0*one(σx())
                    throw("Error two non-touching terms anticommute $i, $j, $k")
                end
            end
        end
    end
end
println(total_pauli_weight/total_hopping_terms)