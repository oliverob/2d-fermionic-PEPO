    using SparseArrayKit, LinearAlgebra, TensorOperations, SparseArrays, TensorCore

    Z = SparseArray{Int8}([1 0; 0 -1])
    X = SparseArray{Int8}([0 1; 1 0])
    creation_operator = SparseArray{Int8}([0 0; 1 0])
    annihilation_operator = SparseArray{Int8}([0 1; 0 0])


    function get_JW_matrix(i, num_of_fermions, filled, creation)
        if i > num_of_fermions
            throw("Not enough space")
        end
        if i <= 0
            throw("Out of range")
        end
        JW_matrix = i == 1 ? I(2) : Z
        JW_matrix = SparseArray{Int8}(JW_matrix)
        if !filled
            vaccum = SparseArray{Int8}([1 0; 0 0])
            @tensor JW_matrix[a,b,c,d,e,f] := SparseArray{Int8}(I(2^(i-1)))[a,d]*vaccum[b,e]*SparseArray{Int8}(I(2^(num_of_fermions-i)))[c,f]
        else
            if i == 1
                if creation
                    @tensor JW_matrix[b,c,e,f] := creation_operator[b,e]*SparseArray{Int8}(I(2^(num_of_fermions-i)))[c,f]
                else
                    @tensor JW_matrix[b,c,e,f] := annihilation_operator[b,e]*SparseArray{Int8}(I(2^(num_of_fermions-i)))[c,f]
                end
            else
                Zs = Z
                for j in 2:i-1
                    @tensor Zs[a,b,c,d] := Zs[a,c]*Z[b,d]
                    Zs = reshape(Zs, 2^(j), 2^(j))
                end
                
                if creation
                    @tensor JW_matrix[a,b,c,d,e,f] := Zs[a,d]*creation_operator[b,e]*SparseArray{Int8}(I(2^(num_of_fermions-i)))[c,f]
                else
                    @tensor JW_matrix[a,b,c,d,e,f] := Zs[a,d]*annihilation_operator[b,e]*SparseArray{Int8}(I(2^(num_of_fermions-i)))[c,f]
                end
            end

        end
        JW_matrix = reshape(JW_matrix, (Int(sqrt(length(JW_matrix))),Int(sqrt(length(JW_matrix)))))

        return JW_matrix
    end
    vertex_tensor(i,j,k, num_of_fermions) = get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,false,true)+ 
                                            get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,false,true)+ 
                                            get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,true,true)+ 
                                            get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,true,true);

    function GHZ_tensor(j,k, num_of_fermions)
        spin_1 = SparseArray{Int8}([1; 0])
        spin_0 = SparseArray{Int8}([0; 1])
        @tensor GHZ_tensor[a,b,c]  := spin_1[a]*(get_JW_matrix(j, num_of_fermions,false, false)*get_JW_matrix(k, num_of_fermions,false,false))[b,c] +
        spin_0[a]*(get_JW_matrix(j, num_of_fermions,true,false)*get_JW_matrix(k, num_of_fermions,true,false))[b,c]
        return GHZ_tensor
    end

    X_tensor(i,num_of_fermions, on) = on ? get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(i, num_of_fermions,true,false) + get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(i, num_of_fermions,false,false) : SparseArray{Int8}(I(2^(num_of_fermions)))
    Z_tensor(i,num_of_fermions, on) = on ? get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(i, num_of_fermions,false,false) - get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(i, num_of_fermions,true,false) : SparseArray{Int8}(I(2^(num_of_fermions)))


    function trace_out_fermion(tensor,is,  num_of_fermions)
        # Should do (0| stuff |0) + (1| stuff |1) for each fermion to trace Out
        # Assumes spin, fermion in, fermion out
        current_tensor = copy(tensor)
        for i in is
            @tensor expectation_0[a,b,c] := get_JW_matrix(i, num_of_fermions,false, false)[b,e]*current_tensor[a,e,f]*get_JW_matrix(i, num_of_fermions,false, true)[f,c]
            @tensor expectation_1[a,b,c] := get_JW_matrix(i, num_of_fermions,true, false)[b,e]*current_tensor[a,e,f]*get_JW_matrix(i, num_of_fermions,true, true)[f,c]
            current_tensor = expectation_0+expectation_1
        end
        return current_tensor
    end

    function multiply_by_vacuum(MPO, num_of_fermions, num_of_physical_fermions, num_of_spins)
        partial_vaccum = SparseArray(reshape([j==((k-1)*2^(num_of_fermions-num_of_physical_fermions))+1 ? 1 : 0 for j in 1:2^(num_of_fermions) for k in 1:2^num_of_physical_fermions],2^num_of_physical_fermions,2^num_of_fermions))
        full_vacuum = SparseArray([j == 1 ? 1 : 0 for j in 1:2^num_of_fermions])
        @tensor MPO_without_virtual_fermions[b,a] := partial_vaccum[b,e]*MPO[a,e,f]*full_vacuum[f] # Finally correct!
        return MPO_without_virtual_fermions
    end


    function get_1D_MPO(num_of_sites,X_defect,Z_defect,shift)
        MPO = vertex_tensor(2,1,2*num_of_sites+((shift)% num_of_sites+1),3*num_of_sites) # 2.5
        # println(2*num_of_sites+((shift)% num_of_sites+1))
        for i in 2:num_of_sites # 7.5
            # println(2*num_of_sites+((i-1+shift) % num_of_sites+1))
            MPO *= vertex_tensor(2*i,2*(i-1)+1,2*num_of_sites+((i-1+shift) % num_of_sites+1),3*num_of_sites)
        end
        
        @tensor MPO[c,a,d] := MPO[a,b]*GHZ_tensor(2*num_of_sites, 1, 3*num_of_sites)[c,b,d] # 0.85

        MPO = reshape(MPO, 2^1,2^(3*num_of_sites), 2^(3*num_of_sites))

        for i in 2:num_of_sites # 2.5
            @tensor MPO[a,c,b,d] := MPO[a,b,e]*GHZ_tensor(2*(i-1), 2*(i-1)+1, 3*num_of_sites)[c,e,d]
            MPO = reshape(MPO, 2^i,2^(3*num_of_sites), 2^(3*num_of_sites))
        end
        MPO = reshape(reshape(MPO, 2^(4*num_of_sites), 2^(3*num_of_sites))*X_tensor(1,3*num_of_sites,X_defect)*Z_tensor(1,3*num_of_sites,Z_defect),2^num_of_sites,2^(3*num_of_sites),2^(3*num_of_sites))

        traced_MPO = multiply_by_vacuum(trace_out_fermion(MPO,[j for j in 1:2*num_of_sites],  3*num_of_sites), 3*num_of_sites, num_of_sites,num_of_sites) # 6.5

        if shift != 0
            # Translate spins
            spins_tensor_product_dims = (2 for j in 1:num_of_sites) |> Tuple
            # println([j== 1 ? 1 : ((j+num_of_sites-shift-2) % num_of_sites)+2 for j in 1:num_of_sites+1] )
            traced_MPO = reshape(permutedims(reshape(traced_MPO, (2^(num_of_sites),spins_tensor_product_dims...)),
            [j== 1 ? 1 : ((j+num_of_sites-shift-2) % num_of_sites)+2 for j in 1:num_of_sites+1])
            ,2^(num_of_sites), 2^(num_of_sites))
        end
        return traced_MPO
    end

    function charge_sectors(MPO, num_of_sites)
        # ZZ Symmetry
        ZZ = Z
        for _ in 2:num_of_sites
            ZZ = ZZ ⊗ Z
        end
        ZZ = reshape(permutedims(ZZ, cat([2*i+1 for i in 0:num_of_sites-1],[2*i for i in 1:num_of_sites],dims=1)),2^num_of_sites, 2^num_of_sites)
        ZZ*MPO == MPO ? println("ZZ: +1") : println("ZZ: -1")
        # XX Symmetry
        XX = X
        for _ in 2:num_of_sites
            XX = XX ⊗ X
        end
        XX = reshape(permutedims(XX, cat([2*i+1 for i in 0:num_of_sites-1],[2*i for i in 1:num_of_sites],dims=1)),2^num_of_sites, 2^num_of_sites)
        MPO*XX == MPO ? println("XX: +1") : println("XX: -1")
    end

    function twists(MPO, translated_MPO, num_of_sites)
        translated_MPO == MPO ? println("Twist: I") :
        reshape(permutedims(Z ⊗I(2^(num_of_sites-1)),[1,3,2,4]),2^(num_of_sites),2^(num_of_sites))*translated_MPO == MPO ? println("Twist: Z") :
        translated_MPO*reshape(permutedims( I(2) ⊗ X ⊗ I(2^(num_of_sites-2))  ,[1,3,5,2,4,6]),2^(num_of_sites),2^(num_of_sites))  == MPO ? println("Twist: X") :
        reshape(permutedims(Z ⊗ I(2^(num_of_sites-1)),[1,3,2,4]),2^(num_of_sites),2^(num_of_sites))*translated_MPO*reshape(permutedims( I(2) ⊗ X ⊗ I(2^(num_of_sites-2)),[1,3,5,2,4,6]),2^(num_of_sites),2^(num_of_sites)) == MPO ? println("Twist: ZX") : println()
    end

    num_of_sites = 5
    for X_defect in (false, true)
        for Z_defect in (false, true)
            MPO = get_1D_MPO(num_of_sites, X_defect, Z_defect,0)

            println("Defect: ", X_defect & Z_defect ? "Y" : X_defect ? "X" : Z_defect ? "Z" : "I")
            charge_sectors(MPO, num_of_sites)

            translated_MPO = get_1D_MPO(num_of_sites, X_defect, Z_defect,1)

            twists(MPO, translated_MPO, num_of_sites)
            
            println()
        end
    end

    # @time get_JW_matrix(1, 15, false, false)    