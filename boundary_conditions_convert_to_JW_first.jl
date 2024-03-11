using TensorCore
using LinearAlgebra

Z = [1 0; 0 -1]
X = [0 1; 1 0]

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

GHZ_tensor(j,k, num_of_fermions) = reshape( [1 0]'⊗(get_JW_matrix(j, num_of_fermions,false, false)*get_JW_matrix(k, num_of_fermions,false,false)) +
                                                            [0 1]'⊗(get_JW_matrix(j, num_of_fermions,true,false)*get_JW_matrix(k, num_of_fermions,true,false)),2,2^(num_of_fermions), 2^(num_of_fermions))

vertex_tensor(i,j,k, num_of_fermions) = reshape((get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,false,true)+ 
                                        get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,false,true)+ 
                                        get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,true,true)+ 
                                        get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,true,true)),2^(num_of_fermions), 2^(num_of_fermions));

X_tensor(i,num_of_fermions, on) = on ? get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(i, num_of_fermions,true,false) + get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(i, num_of_fermions,false,false) : I(2^(num_of_fermions))
Z_tensor(i,num_of_fermions, on) = on ? get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(i, num_of_fermions,false,false) - get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(i, num_of_fermions,true,false) : I(2^(num_of_fermions))

function multiply_by_vacuum(MPO, num_of_fermions, num_of_physical_fermions, num_of_spins)
    reshape(reshape([j==((k-1)*2^(num_of_fermions-num_of_physical_fermions))+1 ? 1 : 0 for j in 1:2^(num_of_fermions) for k in 1:2^num_of_physical_fermions],2^num_of_physical_fermions,2^num_of_fermions)
    *reshape(reshape(MPO,2^(num_of_fermions+num_of_spins), 2^(num_of_fermions))*[j == 1 ? 1 : 0 for j in 1:2^num_of_fermions],2^(num_of_spins),2^(num_of_fermions))',2^(num_of_physical_fermions),2^(num_of_spins)) # Finally correct!
end

function trace_out_fermion(tensor,is,  num_of_fermions, num_of_spins)
    # Should do (0| stuff |0) + (1| stuff |1) for each fermion to trace Out
    # Assumes spin, fermion in, fermion out
    current_tensor = copy(tensor)
    for i in is
        expectation_0 = reshape(permutedims(reshape(get_JW_matrix(i, num_of_fermions,false, false)*reshape(permutedims(reshape(current_tensor*get_JW_matrix(i, num_of_fermions,false, true), 2^num_of_spins, 2^num_of_fermions, 2^num_of_fermions), [2, 3 ,1]), 2^(num_of_fermions), 2^(num_of_fermions+num_of_spins)),  2^num_of_fermions, 2^num_of_fermions, 2^num_of_spins,), [3, 1,2]), 2^(num_of_fermions+num_of_spins), 2^(num_of_fermions))
        expectation_1 = reshape(permutedims(reshape(get_JW_matrix(i, num_of_fermions,true, false)*reshape(permutedims(reshape(current_tensor*get_JW_matrix(i, num_of_fermions,true, true), 2^num_of_spins, 2^num_of_fermions, 2^num_of_fermions), [2, 3 ,1]), 2^(num_of_fermions), 2^(num_of_fermions+num_of_spins)),  2^num_of_fermions, 2^num_of_fermions, 2^num_of_spins,), [3, 1,2]), 2^(num_of_fermions+num_of_spins), 2^(num_of_fermions))
        current_tensor = expectation_0-expectation_1
    end
    return reshape(current_tensor, size(tensor))
end

for X_defect in (false, true)
    for Z_defect in (false, true)
        MPO = reshape(permutedims(reshape(reshape(reshape(GHZ_tensor(1,2,6),2^7,2^6)*X_tensor(2,6,X_defect)*Z_tensor(2,6,Z_defect)*vertex_tensor(2,3,5,6)*reshape(permutedims(GHZ_tensor(3,4,6),[2,1,3]),2^6,2^7), 2^8, 2^6)*vertex_tensor(4,1,6,6),2,2^6,2, 2^6), [1,3,2,4]), 2^8, 2^6)
        traced_MPO = multiply_by_vacuum(trace_out_fermion(MPO,[1,2,3,4],  6, 2), 6, 2, 2)

        println("Defect: ", X_defect & Z_defect ? "Y" : X_defect ? "X" : Z_defect ? "Z" : "I")

        reshape(permutedims(Z ⊗ Z,[1,3,2,4]),4,4)*traced_MPO == traced_MPO ? println("ZZ: +1") : println("ZZ: -1")
        traced_MPO*reshape(permutedims(X ⊗ X,[1,3,2,4]),4,4) == traced_MPO ? println("XX: +1") : println("XX: -1")

        translated_MPO = reshape(permutedims(reshape(reshape(reshape(GHZ_tensor(1,2,6),2^7,2^6)*X_tensor(2,6,X_defect)*Z_tensor(2,6,Z_defect)*vertex_tensor(2,3,6,6)*reshape(permutedims(GHZ_tensor(3,4,6),[2,1,3]),2^6,2^7), 2^8, 2^6)*vertex_tensor(4,1,5,6),2,2^6,2, 2^6), [1,3,2,4]), 2^8, 2^6)
        traced_translated_MPO = reshape(permutedims(reshape(multiply_by_vacuum(trace_out_fermion(translated_MPO,[1,2,3,4],  6, 2), 6, 2, 2),2,2,2,2),[1,2,4,3]),4,4)
        traced_translated_MPO == traced_MPO ? println("Twist: I") :
        reshape(permutedims(Z ⊗ I(2),[1,3,2,4]),4,4)*traced_translated_MPO == traced_MPO ? println("Twist: Z") :
        traced_translated_MPO*reshape(permutedims(I(2)⊗ X,[1,3,2,4]),4,4) == traced_MPO ? println("Twist: X") :
        reshape(permutedims(Z ⊗ I(2),[1,3,2,4]),4,4)*traced_translated_MPO*reshape(permutedims(I(2)⊗ X,[1,3,2,4]),4,4) == traced_MPO ? println("Twist: ZX") : println()
        
        println()
    end
end
