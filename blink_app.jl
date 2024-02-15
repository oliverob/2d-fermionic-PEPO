include("./arbitary_graph_operator_conversion.jl")
using .PEPOAnalysis
using Graphs, GraphPlot
using Blink
using Interact

window = Window()
html = """<sliders></sliders><graphs></graphs><results></results>"""
body!(window, html, async=false)
println()

function get_graphs(interaction_graph_selection,PEPO_graph_selection, num_of_fermions)
    
    interaction_graph = complete_graph(num_of_fermions)
    if interaction_graph_selection == "Grid"
        width = floor(Int, sqrt(num_of_fermions))
        interaction_graph = grid([width,floor(Int, num_of_fermions/width)])
    end
    PEPO_graph = complete_graph(num_of_fermions)
    if PEPO_graph_selection == "Grid"
        width = floor(Int, sqrt(num_of_fermions))
        PEPO_graph = grid([width,floor(Int, num_of_fermions/width)])
        if interaction_graph_selection == "Complete Graph"
            interaction_graph = complete_graph(width*floor(Int, num_of_fermions/width))
                
        end
    elseif PEPO_graph_selection == "MST"
        PEPO_graph = Graph(prim_mst(interaction_graph))
            
    end


    PEPO = convert_to_PEPO(PEPO_graph, false)

    if (!is_connected(PEPO_graph))
        throw("Not connected")
    end
    
    content!(window, "graphs", hbox(gplot(interaction_graph),gplot(PEPO, edgelabel=map((x)-> x[2],sort(collect(PEPO.edge_data), by=x->x[1])), edgelabelc="blue", EDGELABELSIZE=8, nodelabel=map((x)-> x[2][2][1],sort(collect(PEPO.vertex_properties), by=x->x[1])))))
#     gplot(PEPO, edgelabel=map((x)-> x[2],sort(collect(PEPO.edge_data), by=x->x[1])), edgelabelc="blue", EDGELABELSIZE=8, nodelabel=map((x)-> x[2][2][1],sort(collect(PEPO.vertex_properties), by=x->x[1]))))

#     display_graphs(PEPO, PEPO_graph, interaction_graph)
    return PEPO, interaction_graph
end

function compute_average_weights(interaction_graph_selection,PEPO_graph_selection, num_of_fermions)
    PEPO, interaction_graph = get_graphs(interaction_graph_selection,PEPO_graph_selection, num_of_fermions)
    results = ""
    try 
        results *= "<p>Odd algebra average Pauli weight: "* string(testing_odd_algebra(PEPO, interaction_graph)) *"</p>"
    catch (e)
        results *= "<p>FAILED ODD ALGEBRA: $e</p>"
    end

    try 
        results *= "Even algebra average Pauli weight: "* string(testing_even_algebra(PEPO, interaction_graph))
    catch (e)
        results *= "FAILED EVEN ALGEBRA: $e"
    end
    content!(window, "results", results)
end

interaction_graph_selection = dropdown(["Complete Graph","Grid"])
PEPO_graph_selection = dropdown(["Complete Graph","Grid", "MST"])
num_of_fermions = slider(1:10, readout=true)
on(n -> (compute_average_weights(n,PEPO_graph_selection[], num_of_fermions[]);),interaction_graph_selection)
on(n -> (compute_average_weights(interaction_graph_selection[],n, num_of_fermions[])),PEPO_graph_selection)
on(n -> (compute_average_weights(interaction_graph_selection[],PEPO_graph_selection[], n)),num_of_fermions)
content!(window,"sliders", vbox(hbox("Interaction graph: ", interaction_graph_selection),hbox("PEPO graph: ", PEPO_graph_selection),hbox("Number of fermions: ",num_of_fermions)), async=false)
