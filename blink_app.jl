include("./arbitary_graph_operator_conversion.jl")
using .PEPOAnalysis
using Graphs, GraphPlot
using Blink
using Interact
using QuantumAlgebra
function partial_binary_tree(n::T) where {T<:Integer}
    n <= 0 && return SimpleGraph(0)
    n == 1 && return SimpleGraph(1)

    k = ceil(Int,log(2,n))
    print(k)
    ne = Int(n)
    fadjlist = Vector{Vector{T}}(undef, n+1)
    @inbounds fadjlist[1] = T[2]
    @inbounds fadjlist[2] = T[3, 4]
    @inbounds for i in 1:k
        @simd for j in (2^i):min(n,(2^(i + 1) - 1))
            if  2j+2 <= n+1
                fadjlist[j+1] = T[j ÷ 2+1, 2j+1, 2j + 2]
            elseif 2j+1 <= n+1
                fadjlist[j+1] = T[j ÷ 2+1, 2j+1]
            else
                fadjlist[j+1] = T[j ÷ 2+1]
            end
        end
    end
    return SimpleGraph(ne, fadjlist)
end


function get_graphs(interaction_graph_selection,PEPO_graph_selection, num_of_fermions,defect)
    max_fermions = num_of_fermions
    if interaction_graph_selection == "Grid" || PEPO_graph_selection == "Grid" ||interaction_graph_selection == "Torus" ||  PEPO_graph_selection == "Torus" 
        max_fermions = floor(Int,sqrt(num_of_fermions))^2
    end
    interaction_graph = complete_graph(max_fermions)
    if interaction_graph_selection == "Grid" || interaction_graph_selection == "Torus"
        width = floor(Int, sqrt(max_fermions))
        interaction_graph = grid([width,width], periodic=(interaction_graph_selection == "Torus"))
    end
    PEPO_graph = complete_graph(max_fermions)
    if PEPO_graph_selection == "Grid" ||  PEPO_graph_selection == "Torus"
        width = floor(Int, sqrt(max_fermions))
        PEPO_graph = grid([width,width], periodic=(PEPO_graph_selection == "Torus"))
    elseif PEPO_graph_selection == "MST"
        PEPO_graph = Graph(prim_mst(interaction_graph))
    elseif PEPO_graph_selection == "Binary Tree"
        PEPO_graph = partial_binary_tree(max_fermions)
    end


    PEPO = convert_to_PEPO(PEPO_graph, defect[])

    if (!is_connected(PEPO_graph))
        throw("Not connected")
    end
    
    return PEPO, interaction_graph, PEPO_graph
end

function get_results(PEPO, interaction_graph)
    odd_results = ""
    try 
        odd_results *= "Odd algebra average Pauli weight: "* string(testing_odd_algebra(PEPO, interaction_graph)) 
    catch (e)
        odd_results *= "FAILED ODD ALGEBRA: $e"
    end
    even_results = ""
    try 
        even_results *= "Even algebra average Pauli weight: "* string(testing_even_algebra(PEPO, interaction_graph))
    catch (e)
        even_results *= "FAILED EVEN ALGEBRA: $e"
    end
    return vbox(dom"p"(odd_results), dom"p"(even_results))
end

function example_operator(PEPO,interaction_graph,defect)
    map_from_t_to_str = Dict([QuantumAlgebra.TLSx_=> "X",QuantumAlgebra.TLSy_ => "Y", QuantumAlgebra.TLSz_=>"Z"])
    vertex_colors = Vector{String}(undef, nv(PEPO))
    fill!(vertex_colors, "grey")
    vertex_colors[collect(vertices(interaction_graph))] .= "purple"
    operators = 0
    if defect[]
        vertex = rand(collect(vertices(interaction_graph)))
        operators = fermionic_X(PEPO, vertex)
        vertex_colors[vertex] = "black"
    else
        edge = rand(collect(edges(interaction_graph)))
        operators = fermionic_XX(PEPO, src(edge), dst(edge))
        vertex_colors[src(edge)] = "black"
        vertex_colors[dst(edge)] = "black"
    end
    vertex_labels = Vector{String}(undef, nv(PEPO))
    fill!(vertex_labels, "")
    
    for operator in collect(keys(operators.terms))[1].bares.v
        vertex_labels[operator.inds[1].num] = map_from_t_to_str[operator.t]
    end
    for vertex in vertices(PEPO)
        if PEPO[vertex][1] == :d
            vertex_colors[vertex] = "yellow"
        end
    end
    return vertex_labels, vertex_colors
end

function refresh(interaction_graph_selection,PEPO_graph_selection, num_of_fermions, defect)
    PEPO, interaction_graph, PEPO_graph= get_graphs(interaction_graph_selection,PEPO_graph_selection, num_of_fermions, defect)
    results = get_results(PEPO, interaction_graph)
    vertex_label, vertex_colors = example_operator(PEPO,interaction_graph,defect)
    graphs = 
        hbox(
            gplot(interaction_graph),
    # gplot(PEPO, edgelabel=map((x)-> x[2],sort(collect(PEPO.edge_data), by=x->x[1])), edgelabelc="blue", EDGELABELSIZE=8, nodelabel=map((x)-> x[2][2][1],sort(collect(PEPO.vertex_properties), by=x->x[1])))),
    gplot(PEPO, edgelabel=map((x)-> x[2],sort(collect(PEPO.edge_data), by=x->x[1])), edgelabelc="blue", EDGELABELSIZE=8, nodelabel=vertex_label, nodefillc=vertex_colors))
    return vbox(graphs, results)
end

interaction_graph_selection = dropdown(["Complete Graph","Grid", "Torus"])
PEPO_graph_selection = dropdown(["Complete Graph","Grid","Torus", "MST", "Binary Tree"])
num_of_fermions = slider(1:16, readout=true)
defect = checkbox("Defect")

interactive_plot = map(refresh,interaction_graph_selection, PEPO_graph_selection,num_of_fermions,defect)
# window = Window()
ui = vbox(hbox("Interaction graph: ", interaction_graph_selection),
hbox("PEPO graph: ", PEPO_graph_selection),
hbox("Number of fermions: ",num_of_fermions), 
defect, 
interactive_plot)
# body!(window, ui, async=false)
# refresh(interaction_graph_selection[],PEPO_graph_selection[], num_of_fermions[], defect[])
# println("")
using Interact, Mux
WebIO.webio_serve(page("/", req -> ui), 8001) # serve on a random port