using Graphs
using MetaGraphsNext
using GraphPlot, Compose
using QuantumAlgebra

function convert_to_PEPO(G, defect=false)
    new_graph = MetaGraph(DiGraph(), label_type=Int, vertex_data_type=Tuple{Symbol, Int}, edge_data_type=Int)
    for vertex in vertices(G)
        new_graph[vertex] = (:v, 1)
    end
    current_edges = edges(G)
    for (i, edge) in enumerate(current_edges)
        u, v = src(edge), dst(edge)
        if i == 1 && defect
            new_graph[nv(new_graph)+1] = (:d,0)
            new_graph[u, nv(new_graph)] = new_graph[u][2]
        else
            new_graph[nv(new_graph)+1] = (:e,0)
            new_graph[u, nv(new_graph)] = new_graph[u][2]
        end
        new_graph[u] = (:v, new_graph[u][2]+1)
        new_graph[nv(new_graph), v] =  new_graph[v][2]
        new_graph[v] = (:v, new_graph[v][2]+1)
    end
    
    return new_graph
end

function has_order_higher_than_path(PEPO,source,path, leg)
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
        z_edges_src = filter((x)-> has_order_higher_than_path(PEPO,source,path, x),src_neighbours)
        z_edges_dst = filter((x)-> has_order_higher_than_path(PEPO,destination,path, x),dest_neighbours)
        if PEPO[path][1] == :d
            operators *= σz(path)
            break;
        end
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


function fermionic_X(PEPO,target)
    
    defect = filter((x)-> PEPO[x][1] == :d,collect(vertices(PEPO)))
    if length(defect) == 0
        throw("No defect")
    end
    shortest_path = a_star(Graph(PEPO), target, defect[1])

    operators = σz()^2
    for edge in shortest_path
        source = src(edge) 
        destination = dst(edge) 
        vertex_tensor = PEPO[source][1] == :v ? source : destination
        ghz_tensor = PEPO[source][1] == :v ? destination : source
        vertex_neighbours = all_neighbors(PEPO, vertex_tensor)
        z_edges_vertex = filter((x)-> has_order_higher_than_path(PEPO,vertex_tensor,ghz_tensor, x),vertex_neighbours)
        operators *= prod(map((x)-> σz(x),z_edges_vertex))

        if PEPO[ghz_tensor][1] == :d && has_edge(PEPO, vertex_tensor, ghz_tensor)
            operators *= σz(ghz_tensor)
            break;
        end
        if destination == ghz_tensor
            operators *= has_edge(PEPO,vertex_tensor, ghz_tensor) ? 1 : -1
            operators *= σy(ghz_tensor)
        end
        if PEPO[ghz_tensor][1] == :d
            break;
        end
    end
    return normal_form(operators)
end

l = 4
G = smallgraph(:petersen)
# G = Graph(prim_mst(complete_graph(10)))
# G = complete_graph(20) 

PEPO = convert_to_PEPO(G, true)

if (!is_connected(G))
    throw("Not connected")
end

t = gplot(PEPO, edgelabel=map((x)-> x[2],sort(collect(PEPO.edge_data), by=x->x[1])), edgelabelc="white", EDGELABELSIZE=8, nodelabel=map((x)-> x[2][2][1],sort(collect(PEPO.vertex_properties), by=x->x[1])))

t = gplot(PEPO, edgelabel=map((x)-> x[2],sort(collect(PEPO.edge_data), by=x->x[1])), edgelabelc="white", EDGELABELSIZE=8, nodelabel=map((x)-> x[2][1],sort(collect(PEPO.vertex_properties), by=x->x[1])))

gplot(G)

fermionic_XX(PEPO,1, 2)


# Testing even fermionic algebra
function testing_even_algebra(PEPO, G)
    total_pauli_weight = 0
    total_hopping_terms = 0
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
    return total_pauli_weight/total_hopping_terms
end


# Testing even fermionic algebra
function testing_odd_algebra(PEPO, G)
    total_pauli_weight = 0
    total_hopping_terms = 0
    for i in 1:nv(G)
        current_majorana_term = fermionic_X(PEPO,i) 
        total_pauli_weight += length(collect(keys(current_majorana_term.terms))[1].bares.v)
        total_hopping_terms += 1
        for j in i:nv(G)
                other_majorana_term = fermionic_X(PEPO, j)
                if (i != j)
                    if normal_form(comm(current_majorana_term, other_majorana_term)) == 0*one(σx())
                        throw("Error two different majoranas commute: $i, $j, $current_majorana_term, $other_majorana_term")
                    end
                if i == j
                    if normal_form(comm(current_majorana_term, other_majorana_term)) != 0*one(σx())
                        throw("Error identical majoranas anticommute $i, $j")
                    end
                end
            end
        end
    end
    return total_pauli_weight/total_hopping_terms
end

testing_odd_algebra(PEPO, G)