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

function get_spin_matrix(i, num_of_spins, filled)
    if i > num_of_spins
        throw("Not enough space")
    end
    if i <= 0
        throw("Out of range")
    end
    spins_matrix = I(2)
    site_matrix = filled ? [0 0; 0 1] : [1 0; 0 0]

    if i == 1
        spins_matrix = site_matrix
        for j in 2:num_of_spins
            spins_matrix=spins_matrix⊗I(2)
        end
    else
        spins_matrix= I(2)
        for j in 2:i-1
            spins_matrix=spins_matrix⊗I(2)
        end
        spins_matrix = spins_matrix⊗site_matrix
        for j in i+1:num_of_spins
            spins_matrix=spins_matrix⊗I(2)
        end
    end

    odd = [2*i+1 for i in 0:num_of_spins-1]
    even = [2*i for i in 1:num_of_spins]
    spins_matrix= permutedims(spins_matrix, cat(odd,even, dims=1))
    spins_matrix = reshape(spins_matrix, (Int(sqrt(length(spins_matrix))),Int(sqrt(length(spins_matrix)))))
    return spins_matrix
end


GHZ_tensor(i,j,k, num_of_fermions, num_of_spins) = reshape( [j == 1 ? 1 : 0 for j in 1:2^num_of_spins]⊗(get_JW_matrix(j, num_of_fermions,false, false)*get_JW_matrix(k, num_of_fermions,false,false)) +
                                                            [j == i+1 ? 1 : 0 for j in 1:2^num_of_spins]⊗(get_JW_matrix(j, num_of_fermions,true,false)*get_JW_matrix(k, num_of_fermions,true,false)),2^(num_of_fermions+num_of_spins), 2^(num_of_fermions))

vertex_tensor(i,j,k, num_of_fermions, num_of_spins) = reshape((get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,false,true)+ 
                                        get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,false,true)+ 
                                        get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,true,true)+ 
                                        get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,true,true)),2^(num_of_fermions), 2^(num_of_fermions));

X_tensor(i,num_of_fermions,num_of_spins, on) = on ? reshape(permutedims(I(2^(num_of_spins))⊗(get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(i, num_of_fermions,true,false) + get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(i, num_of_fermions,false,false)),[1,3,2,4]),2^(num_of_fermions+num_of_spins),2^(num_of_fermions+num_of_spins)) : I(2^(num_of_fermions+num_of_spins))
Z_tensor(i,num_of_fermions,num_of_spins, on) = on ? reshape(permutedims(I(2^(num_of_spins))⊗(get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(i, num_of_fermions,false,false) - get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(i, num_of_fermions,true,false)),[1,3,2,4]),2^(num_of_fermions+num_of_spins),2^(num_of_fermions+num_of_spins)) : I(2^(num_of_fermions+num_of_spins))
# [i == 1 ? 1 : 0 for i in 1:2^(3)]'*reshape(vertex_tensor(1,2,3,3,1),8,32)*reshape(GHZ_tensor(1,1,2,3,1),32,8)*[i == 1 ? 1 : 0 for i in 1:2^(3)] 
# reshape(reshape(GHZ_tensor(1,2,1,3,1),32,8)*[i == 1 ? 1 : 0 for i in 1:2^(3)],2, 16)*reshape([i == 1 ? 1 : 0 for i in 1:2^(3)]'*reshape(vertex_tensor(2,1,3,3,1),8,32),16,2)

# reshape(reshape(GHZ_tensor(1,1,2,3,1),32,8)*[i == 1 ? 1 : 0 for i in 1:2^(3)],15, 16)
# reshape([i == 1 ? 1 : 0 for i in 1:2^(3)]'*reshape(vertex_tensor(2,1,3,3,1),8,32),16,2)
reshape(GHZ_tensor(1,1,2,3,1)*vertex_tensor(2,1,3,3,1),2,8,8)[2,:,:]

GHZ_tensor(1,1,2,3,1)*[1 0 0 0 0 0 0 0]'

function multiple_by_vacuum(MPO, num_of_fermions, num_of_physical_fermions, num_of_spins)
    reshape([j == 1 ? 1 : 0 for j in 1:2^(num_of_fermions-num_of_physical_fermions)]'*reshape(permutedims(reshape(MPO*[j == 1 ? 1 : 0 for j in 1:2^num_of_fermions],2^(num_of_spins),2^(num_of_fermions)),[2,1]),2^(num_of_fermions-num_of_physical_fermions),2^(num_of_spins + num_of_physical_fermions)),2^(num_of_physical_fermions),2^(num_of_spins)) # Finally correct!
end

MPO = GHZ_tensor(1,1,2,3,1)*vertex_tensor(2,1,3,3,1)
multiple_by_vacuum(MPO, 3, 1, 1)

# reshape(GHZ_tensor(1,1,2,2,1),16,4)

function trace_out_fermion(tensor,is,  num_of_fermions, num_of_spins)
    # Should do (0| stuff |0) + (1| stuff |1) for each fermion to trace Out
    # Assume fermions in are left dims and fermions out are at right dims
    current_tensor = copy(tensor)
    for i in is
        expectation_0 = reshape(permutedims(I(2^(num_of_spins))⊗ get_JW_matrix(i, num_of_fermions,false, true),[1,3,2,4]),2^(num_of_fermions+num_of_spins), 2^(num_of_fermions+num_of_spins))*current_tensor*reshape(permutedims(I(2^(num_of_spins))⊗ get_JW_matrix(i, num_of_fermions,false, false),[1,3,2,4]),2^(num_of_fermions+num_of_spins), 2^(num_of_fermions+num_of_spins))
        expectation_1 = reshape(permutedims(I(2^(num_of_spins))⊗ get_JW_matrix(i, num_of_fermions,true, true),[1,3,2,4]),2^(num_of_fermions+num_of_spins), 2^(num_of_fermions+num_of_spins))*current_tensor*reshape(permutedims(I(2^(num_of_spins))⊗ get_JW_matrix(i, num_of_fermions,true, false),[1,3,2,4]),2^(num_of_fermions+num_of_spins), 2^(num_of_fermions+num_of_spins))
        current_tensor = expectation_0+expectation_1
    end
    return reshape(current_tensor, size(tensor))
end

MPO = GHZ_tensor(1,2,3,3,1)*vertex_tensor(3,2,1,3,1)
MPO_on_vaccum = reshape(reshape(MPO,2^5,2^3)*[i == 1 ? 1 : 0 for i in 1:2^(3)],2)
MPO_traced = reshape(trace_out_fermion(MPO,[2,3], 3,1),2, 2, 2^2, 2, 2, 2^2) # spin in, fermions in, spin out, fermions out
reshape(MPO_traced,16,16)

# @show MPO_traced[:,4,1,:,1,1] # Maps a_1^{\dagger} a_2^{\dagger}
# @show MPO_traced[:,3,1,:,1,1]# Maps a_2^{\dagger}
@show MPO_traced[:,2,1,:,1,1] # Maps a_1^{\dagger}
@show MPO_traced[:,1,1,:,1,1] # Maps vaccum


reshape(MPO_traced,16,16)
modified_MPO = reshape(reshape(permutedims(X⊗ I(2^3),[1,3,2,4]),2^4,2^4)*reshape(MPO_traced,2^4,2^4),2,2,2^2,2,2,2^2)
reshape(modified_MPO,16,16)

# @show modified_MPO[:,4,1,:,1,1] # Maps a_1^{\dagger} a_2^{\dagger}
# @show modified_MPO[:,3,1,:,1,1]# Maps a_2^{\dagger}
@show modified_MPO[:,2,1,:,1,1] # Maps a_1^{\dagger}
@show modified_MPO[:,1,1,:,1,1] # Maps vaccum
@show reshape(reshape(MPO_traced,2^8,2^8)*Z_tensor(1,6,2,true)*Z_tensor(2,6,2,true),2^8,2^8) == reshape(MPO_traced,2^8,2^8)

println()