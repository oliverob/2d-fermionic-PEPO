### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ e66a836f-d4b8-4e74-b0a1-b0e3fc247fdb
# ╠═╡ show_logs = false
import Pkg;Pkg.activate(".");

# ╔═╡ e8cefc27-3c69-435d-a9cf-74ac7c785bc9
using Graphs, GraphPlot, Compose, PlutoUI, QuantumAlgebra

# ╔═╡ 2cfc8f8e-f028-4f4f-ab80-70330e380d5a
include("./arbitary_graph_operator_conversion.jl")

# ╔═╡ 1121856e-abdb-4127-b4fa-0ae61fc53ca8
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


# ╔═╡ f696fbef-9293-4b3a-bd78-33b8e2fdcc57


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


    PEPO = PEPOAnalysis.convert_to_PEPO(PEPO_graph, defect[])

    if (!is_connected(PEPO_graph))
        throw("Not connected")
    end
    
    return PEPO, interaction_graph, PEPO_graph
end


# ╔═╡ d2dd858c-b983-476a-b10f-89ca16893fd3
function get_results(PEPO, interaction_graph)
    results = ""
    try 
        results *= "Odd algebra average Pauli weight: "* string(PEPOAnalysis.testing_odd_algebra(PEPO, interaction_graph)) *"\n"
    catch (e)
        results *= "FAILED ODD ALGEBRA: $e\n"
    end
    try 
        results *= "Even algebra average Pauli weight: "* string(PEPOAnalysis.testing_even_algebra(PEPO, interaction_graph))
    catch (e)
        results *= "FAILED EVEN ALGEBRA: $e"
    end
    return results
end

# ╔═╡ 33cbb434-5d69-4350-9e1c-d58033a5b7b0
function example_operator(PEPO,interaction_graph,defect)
    map_from_t_to_str = Dict([QuantumAlgebra.TLSx_=> "X",QuantumAlgebra.TLSy_ => "Y", QuantumAlgebra.TLSz_=>"Z"])
    vertex_colors = Vector{String}(undef, nv(PEPO))
    fill!(vertex_colors, "grey")
    vertex_colors[collect(vertices(interaction_graph))] .= "purple"
    operators = 0
    if defect[]
        vertex = rand(collect(vertices(interaction_graph)))
        operators = PEPOAnalysis.fermionic_X(PEPO, vertex)
        vertex_colors[vertex] = "black"
    else
        edge = rand(collect(edges(interaction_graph)))
        operators = PEPOAnalysis.fermionic_XX(PEPO, src(edge), dst(edge))
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

# ╔═╡ 165753ac-c6b3-46a7-b3b4-36a77904f772
function compute_average_weights(interaction_graph_selection,PEPO_graph_selection, num_of_fermions, defect)
    PEPO, interaction_graph, PEPO_graph= get_graphs(interaction_graph_selection,PEPO_graph_selection, num_of_fermions, defect)
    results = get_results(PEPO, interaction_graph)
    vertex_label, vertex_colors = example_operator(PEPO,interaction_graph,defect)
	return results, PEPO, interaction_graph
end

# ╔═╡ 1ca011df-1af9-4b10-9d22-520a1e6b2d74
@bind number_of_fermions Slider(5:15)

# ╔═╡ 1e4973ac-d31a-4350-8750-7d2f425b0806
@bind interaction_graph_selection Select(["Complete Graph","Grid", "Torus"])

# ╔═╡ 6fd7a28b-13ed-4b98-9c2d-144300dc366d
@bind PEPO_graph_selection Select(["Complete Graph","Grid","Torus", "MST", "Binary Tree"])

# ╔═╡ 19237e3f-6a1d-4a83-baa2-f16925a493b8
@bind defect CheckBox()

# ╔═╡ da619173-2914-44da-bccf-f5b290b67bbe
begin
	results, PEPO,  interaction_graph  = compute_average_weights(interaction_graph_selection,PEPO_graph_selection, number_of_fermions, defect)
	Text(results)
end

# ╔═╡ a01aaa7c-99b1-4901-8d77-4ca3bd1d5a09
gplot(interaction_graph)

# ╔═╡ 6b9d45fb-7c27-4586-b896-65c9ce08741c
gplot(PEPO, edgelabel=map((x)-> x[2],sort(collect(PEPO.edge_data), by=x->x[1])), edgelabelc="blue", EDGELABELSIZE=8, nodelabel=map((x)-> x[2][2][1],sort(collect(PEPO.vertex_properties), by=x->x[1])))

# ╔═╡ Cell order:
# ╟─e66a836f-d4b8-4e74-b0a1-b0e3fc247fdb
# ╟─2cfc8f8e-f028-4f4f-ab80-70330e380d5a
# ╟─e8cefc27-3c69-435d-a9cf-74ac7c785bc9
# ╟─1121856e-abdb-4127-b4fa-0ae61fc53ca8
# ╟─f696fbef-9293-4b3a-bd78-33b8e2fdcc57
# ╟─d2dd858c-b983-476a-b10f-89ca16893fd3
# ╟─33cbb434-5d69-4350-9e1c-d58033a5b7b0
# ╠═165753ac-c6b3-46a7-b3b4-36a77904f772
# ╠═1ca011df-1af9-4b10-9d22-520a1e6b2d74
# ╠═1e4973ac-d31a-4350-8750-7d2f425b0806
# ╠═6fd7a28b-13ed-4b98-9c2d-144300dc366d
# ╠═19237e3f-6a1d-4a83-baa2-f16925a493b8
# ╠═da619173-2914-44da-bccf-f5b290b67bbe
# ╠═a01aaa7c-99b1-4901-8d77-4ca3bd1d5a09
# ╠═6b9d45fb-7c27-4586-b896-65c9ce08741c
