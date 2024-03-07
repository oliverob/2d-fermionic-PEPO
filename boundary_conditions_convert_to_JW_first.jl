using TensorCore, Einsum
using LinearAlgebra

Z = [1 0; 0 -1]
X = [0 1; 1 0]
Y = [0 1; -1 0]

function get_JW_matrix(i, num_of_fermions, filled, creation)
    if i > num_of_fermions
        throw("Not enough space")
    end
    if i <= 0
        throw("Out of range")
    end
    JW_matrix = i == 1 ? I(2) : Z

    if !filled
        if i == 1
            JW_matrix = [1 0; 0 0]
            for j in 2:num_of_fermions
                JW_matrix=JW_matrix⊗I(2)
            end
        else
            JW_matrix= I(2)
            for j in 2:i-1
                JW_matrix=JW_matrix⊗I(2)
            end
            JW_matrix = JW_matrix⊗[1 0; 0 0]
            for j in i+1:num_of_fermions
                JW_matrix=JW_matrix⊗I(2)
            end
        end
    else
        for j in 2:i-1
            JW_matrix = JW_matrix⊗Z
        end
        if creation
            JW_matrix = i == 1 ? [0 0; 1 0] :  JW_matrix⊗[0 0; 1 0]
        else
            JW_matrix = i == 1 ? [0 1; 0 0] :  JW_matrix⊗[0 1; 0 0]
        end
        for _ in i+1:num_of_fermions
            JW_matrix = JW_matrix⊗I(2)
        end
    end
    odd = [2*i+1 for i in 0:num_of_fermions-1]
    even = [2*i for i in 1:num_of_fermions]
    JW_matrix= permutedims(JW_matrix, cat(odd,even, dims=1))
    JW_matrix = reshape(JW_matrix, (Int(sqrt(length(JW_matrix))),Int(sqrt(length(JW_matrix)))))

    return  JW_matrix
end
get_JW_matrix(2, 3, true, false)*get_JW_matrix(1, 3, true, false)*get_JW_matrix(3, 3, true, false)

GHZ_tensor(i,j, num_of_fermions) = permutedims([1 0; 0 0]⊗(get_JW_matrix(i, num_of_fermions,false, false)*get_JW_matrix(j, num_of_fermions,false,false))+
                                               [0 0; 0 1]⊗(get_JW_matrix(i, num_of_fermions,true,false)*get_JW_matrix(j, num_of_fermions,true,false)),[1,3,2,4])

vertex_tensor(i,j,k, num_of_fermions) = get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,false,true)+ 
                                        get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,false,true)+ 
                                        get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,true,true)+ 
                                        get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,true,true);

X_tensor(i,num_of_fermions, on) = on ? get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(i, num_of_fermions,true,false) + get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(i, num_of_fermions,false,false) : I(2^(num_of_fermions))
Z_tensor(i,num_of_fermions, on) = on ? get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(i, num_of_fermions,false,false) - get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(i, num_of_fermions,true,false) : I(2^(num_of_fermions))

vertex_tensor(2,1,3,3)
reshape(GHZ_tensor(1,2,3),16,16)

function trace_out_fermion(tensor,is,  num_of_fermions)
    # Should do (0| stuff |0) + (1| stuff |1) for each fermion to trace Out
    # Assume fermions in are left dims and fermions out are at right dims
    current_tensor = copy(tensor)
    for i in is
        expectation_0 = reshape(get_JW_matrix(i, num_of_fermions,false, false)*reshape(current_tensor, 2^num_of_fermions,Int(length(tensor)/2^(num_of_fermions))), Int(length(tensor)/2^(num_of_fermions)),2^num_of_fermions)*get_JW_matrix(i, num_of_fermions,false, true)
        expectation_1 = reshape(get_JW_matrix(i, num_of_fermions,true, false)*reshape(current_tensor, 2^num_of_fermions,Int(length(tensor)/2^(num_of_fermions))), Int(length(tensor)/2^(num_of_fermions)),2^num_of_fermions)*get_JW_matrix(i, num_of_fermions,true, true)
        current_tensor = expectation_0+expectation_1
    end
    return reshape(current_tensor, size(tensor))
end

MPO1 = reshape(reshape(GHZ_tensor(3,4,6),2^8,2^6)*vertex_tensor(4,5,1,6),2,2^6,2,2^6)
MPO2 = reshape(reshape(GHZ_tensor(5,6,6),2^8,2^6)*vertex_tensor(6,3,2,6),2,2^6,2,2^6)

MPO = reshape(reshape(permutedims(MPO1,[2,1,3,4]),2^8,2^6)*X_tensor(3,6, false)*Z_tensor(3,6,false)*reshape(permutedims(MPO2,[2,1,3,4]),2^6,2^8),2^6,2,2,2,2,2^6)

MPO_traced =reshape(permutedims(trace_out_fermion(MPO,[3,4,5,6], 6),[1,2,4,6,3,5]),4,2^4,2,2,4,2^4,2,2)[:,1,:,:,:,1,:,:]

@show reshape(MPO_traced[4,:,:,1,:,:],4,4) # Maps a_1^{\dagger} a_2^{\dagger}
@show reshape(MPO_traced[3,:,:,1,:,:],4,4) # Maps a_2^{\dagger}
@show reshape(MPO_traced[2,:,:,1,:,:],4,4) # Maps a_1^{\dagger}
@show reshape(MPO_traced[1,:,:,1,:,:],4,4) # Maps vaccum


reshape(MPO_traced,16,16)

@show reshape(MPO_traced,16,16)*reshape(permutedims(I(4)⊗X ⊗ X,[1,3,5,2,4,6]),16,16)
@show reshape(Z_tensor(1,2,false)*Z_tensor(2,2,false)*reshape(MPO_traced,4,64),16,16) == reshape(MPO_traced,16,16)

println()