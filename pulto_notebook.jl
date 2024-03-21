### A Pluto.jl notebook ###
# v0.19.38

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

# ╔═╡ e8cefc27-3c69-435d-a9cf-74ac7c785bc9
using Graphs, GraphPlot, Compose, PlutoUI, QuantumAlgebra, MetaGraphsNext


# ╔═╡ 2cfc8f8e-f028-4f4f-ab80-70330e380d5a
begin
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
	
	function fermionic_XX(PEPO, start, target)
	    shortest_path = a_star(Graph(PEPO), start, target)
	    operators= fermionic_path(PEPO,shortest_path)
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
	    operators= fermionic_path(PEPO,shortest_path)
	    return normal_form(operators)
	end
	
	function fermionic_path(PEPO, shortest_path)
	
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
	    return operators
	end
	
	
	# Testing even fermionic algebra
	function testing_even_algebra(PEPO, interaction_graph)
	    total_pauli_weight = 0
	    total_hopping_terms = 0
	    for current_edge in edges(interaction_graph)
	        i, j = src(current_edge), dst(current_edge)
	        current_hopping_term = fermionic_XX(PEPO, i, j) 
	        total_pauli_weight += length(collect(keys(current_hopping_term.terms))[1].bares.v)
	        total_hopping_terms += 1
	        for other_edge in edges(interaction_graph)
	            k, l = src(other_edge), dst(other_edge)
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
	    return total_pauli_weight/total_hopping_terms
	end
	
	
	# Testing even fermionic algebra
	function testing_odd_algebra(PEPO, interaction_graph)
	    total_pauli_weight = 0
	    total_hopping_terms = 0
	    for edge in edges(interaction_graph)
	        i, j = src(edge), dst(edge)
	        current_majorana_term = fermionic_X(PEPO,i) 
	        total_pauli_weight += length(collect(keys(current_majorana_term.terms))[1].bares.v)
	        total_hopping_terms += 1
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
	    return total_pauli_weight/total_hopping_terms
	end
	
	function display_graphs(PEPO, G, interaction_graph)
	    # display(gplot(G))
	    println("Interaction Graph")
	    display(gplot(interaction_graph))
	
	    println("PEPO")
	    display(gplot(PEPO, edgelabel=map((x)-> x[2],sort(collect(PEPO.edge_data), by=x->x[1])), edgelabelc="blue", EDGELABELSIZE=8, nodelabel=map((x)-> x[2][2][1],sort(collect(PEPO.vertex_properties), by=x->x[1]))))
	    
	    # display(gplot(PEPO, edgelabel=map((x)-> x[2],sort(collect(PEPO.edge_data), by=x->x[1])), edgelabelc="white", EDGELABELSIZE=8, nodelabel=map((x)-> x[2][1],sort(collect(PEPO.vertex_properties), by=x->x[1]))))
	end
	
	
	
	
	
end

# ╔═╡ 1121856e-abdb-4127-b4fa-0ae61fc53ca8
function partial_binary_tree(n::T) where {T<:Integer}
    n <= 0 && return SimpleGraph(0)
    n == 1 && return SimpleGraph(1)

    k = ceil(Int,log(2,n))
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
    if interaction_graph_selection == "Square Lattice" || PEPO_graph_selection == "Square Lattice" ||interaction_graph_selection == "Torus" ||  PEPO_graph_selection == "Square Lattice Torus" 
        max_fermions = floor(Int,sqrt(num_of_fermions))^2
    end
    interaction_graph = complete_graph(max_fermions)
    if interaction_graph_selection == "Square Lattice" || interaction_graph_selection == "Square Lattice Torus"
        width = floor(Int, sqrt(max_fermions))
        interaction_graph = grid([width,width], periodic=(interaction_graph_selection == "Square Lattice Torus"))
    end
    PEPO_graph = complete_graph(max_fermions)
    if PEPO_graph_selection == "Square Lattice" ||  PEPO_graph_selection == "Square Lattice Torus"
        width = floor(Int, sqrt(max_fermions))
        PEPO_graph = grid([width,width], periodic=(PEPO_graph_selection == "Square Lattice Torus"))
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


# ╔═╡ d2dd858c-b983-476a-b10f-89ca16893fd3
function get_results(PEPO, interaction_graph)
    results = ""
    try 
        results *= "Odd algebra average Pauli weight: "* string(testing_odd_algebra(PEPO, interaction_graph)) *"\n"
    catch (e)
        results *= "FAILED ODD ALGEBRA: $e\n"
    end
    try 
        results *= "Even algebra average Pauli weight: "* string(testing_even_algebra(PEPO, interaction_graph))
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

# ╔═╡ 165753ac-c6b3-46a7-b3b4-36a77904f772
function compute_average_weights(interaction_graph_selection,PEPO_graph_selection, num_of_fermions, defect)
    PEPO, interaction_graph, PEPO_graph= get_graphs(interaction_graph_selection,PEPO_graph_selection, num_of_fermions, defect)
    results = get_results(PEPO, interaction_graph)
    vertex_label, vertex_colors = example_operator(PEPO,interaction_graph,defect)
	return results, PEPO, interaction_graph,  vertex_label, vertex_colors
end

# ╔═╡ 1ca011df-1af9-4b10-9d22-520a1e6b2d74
md""" Max number of fermions: $(@bind number_of_fermions Slider(4:16, show_value=true, default=10))  """

# ╔═╡ 1e4973ac-d31a-4350-8750-7d2f425b0806
md""" Interaction Graph: $(@bind interaction_graph_selection Select(["Complete Graph","Square Lattice", "Square Lattice Torus"], default="Square Lattice"))"""


# ╔═╡ 6fd7a28b-13ed-4b98-9c2d-144300dc366d
md""" PEPO Graph: $(@bind PEPO_graph_selection Select(["Complete Graph","Square Lattice","Square Lattice Torus", "MST", "Binary Tree"], default="Square Lattice")))"""

# ╔═╡ 19237e3f-6a1d-4a83-baa2-f16925a493b8
md"""Defect: $(@bind defect CheckBox()) """

# ╔═╡ da619173-2914-44da-bccf-f5b290b67bbe
begin
	results, PEPO,  interaction_graph,  vertex_label, vertex_colors  = compute_average_weights(interaction_graph_selection,PEPO_graph_selection, number_of_fermions, defect)
	Text(results)
end

# ╔═╡ 6b9d45fb-7c27-4586-b896-65c9ce08741c
md"""PEPO Graph:

    $(gplot(PEPO, edgelabel=map((x)-> x[2],sort(collect(PEPO.edge_data), by=x->x[1])), edgelabelc="blue", EDGELABELSIZE=8, nodelabel=vertex_label, nodefillc=vertex_colors))"""

# ╔═╡ a01aaa7c-99b1-4901-8d77-4ca3bd1d5a09
md"""Interaction Graph:

$(gplot(interaction_graph))"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Compose = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
GraphPlot = "a2cc645c-3eea-5389-862e-a155d0052231"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
MetaGraphsNext = "fa8bd995-216d-47f1-8a91-f3b68fbeb377"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
QuantumAlgebra = "1b9008d5-2a9d-4901-8cfb-44bb87795b64"

[compat]
Compose = "~0.9.5"
GraphPlot = "~0.5.2"
Graphs = "~1.9.0"
MetaGraphsNext = "~0.7.0"
PlutoUI = "~0.7.57"
QuantumAlgebra = "~1.3.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "5929a1fef86a38abed10ee9400d41338588f4f14"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "c278dfab760520b8bb7e9511b968bf4ba38b7acc"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "d2c021fbdde94f6cdaa799639adfeeaa17fd67f5"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.13.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "bf6570a34c850f99407b494757f5d7ad233a7257"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.5"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "ac67408d9ddf207de5cfa9a97e114352430f01ed"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.16"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "c5c28c245101bd59154f649e19b038d15901b5dc"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.GraphPlot]]
deps = ["ArnoldiMethod", "ColorTypes", "Colors", "Compose", "DelimitedFiles", "Graphs", "LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "5cd479730a0cb01f880eff119e9803c13f214cab"
uuid = "a2cc645c-3eea-5389-862e-a155d0052231"
version = "0.5.2"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "899050ace26649433ef1af25bc17a815b3db52b7"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.9.0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.JLD2]]
deps = ["FileIO", "MacroTools", "Mmap", "OrderedCollections", "Pkg", "PrecompileTools", "Printf", "Reexport", "Requires", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "5ea6acdd53a51d897672edb694e3cc2912f3f8a7"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.4.46"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MetaGraphsNext]]
deps = ["Graphs", "JLD2", "SimpleTraits"]
git-tree-sha1 = "a385fe5aa1384647e55c0c8773457b71e9b08518"
uuid = "fa8bd995-216d-47f1-8a91-f3b68fbeb377"
version = "0.7.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "a6783c887ca59ce7e97ed630b74ca1f10aefb74d"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.57"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuantumAlgebra]]
deps = ["Combinatorics", "LaTeXStrings", "Latexify", "PrecompileTools", "Preferences", "Printf", "REPL"]
git-tree-sha1 = "c55f6cf07b1a611ec15607e440edc0db47009478"
uuid = "1b9008d5-2a9d-4901-8cfb-44bb87795b64"
version = "1.3.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "7b0e9c14c624e435076d19aea1e5cbdec2b9ca37"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.2"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "54194d92959d8ebaa8e26227dbe3cdefcdcd594f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.3"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─e8cefc27-3c69-435d-a9cf-74ac7c785bc9
# ╟─2cfc8f8e-f028-4f4f-ab80-70330e380d5a
# ╟─1121856e-abdb-4127-b4fa-0ae61fc53ca8
# ╟─f696fbef-9293-4b3a-bd78-33b8e2fdcc57
# ╟─d2dd858c-b983-476a-b10f-89ca16893fd3
# ╟─33cbb434-5d69-4350-9e1c-d58033a5b7b0
# ╟─165753ac-c6b3-46a7-b3b4-36a77904f772
# ╟─1ca011df-1af9-4b10-9d22-520a1e6b2d74
# ╟─1e4973ac-d31a-4350-8750-7d2f425b0806
# ╟─6fd7a28b-13ed-4b98-9c2d-144300dc366d
# ╟─19237e3f-6a1d-4a83-baa2-f16925a493b8
# ╟─da619173-2914-44da-bccf-f5b290b67bbe
# ╟─6b9d45fb-7c27-4586-b896-65c9ce08741c
# ╟─a01aaa7c-99b1-4901-8d77-4ca3bd1d5a09
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
