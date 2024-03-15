    using SparseArrayKit, LinearAlgebra, TensorOperations, SparseArrays, TensorCore, Profile, JET

    const Z = SparseArray{Int8}([1 0; 0 -1])
    const X = SparseArray{Int8}([0 1; 1 0])
    const creation_operator = SparseArray{Int8}([0 0; 1 0])
    const annihilation_operator = SparseArray{Int8}([0 1; 0 0])


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
            @tensor JW_matrix[a,b,c,d,e,f] := SparseArray{Int8}(sparse(I(2^(i-1))))[a,d]*vaccum[b,e]*SparseArray{Int8}(sparse(I(2^(num_of_fermions-i))))[c,f]
        else
            if i == 1
                if creation
                    @tensor JW_matrix[b,c,e,f] := creation_operator[b,e]*SparseArray{Int8}(sparse(I(2^(num_of_fermions-i))))[c,f]
                else
                    @tensor JW_matrix[b,c,e,f] := annihilation_operator[b,e]*SparseArray{Int8}(sparse(I(2^(num_of_fermions-i))))[c,f]
                end
            else
                Zs = Z
                for j in 2:i-1
                    @tensor Zs[a,b,c,d] := Zs[a,c]*Z[b,d]
                    Zs = reshape(Zs, 2^(j), 2^(j))
                end
                
                if creation
                    @tensor JW_matrix[a,b,c,d,e,f] := Zs[a,d]*creation_operator[b,e]*SparseArray{Int8}(sparse(I(2^(num_of_fermions-i))))[c,f]
                else
                    @tensor JW_matrix[a,b,c,d,e,f] := Zs[a,d]*annihilation_operator[b,e]*SparseArray{Int8}(sparse(I(2^(num_of_fermions-i))))[c,f]
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
    function vertex_tensor_5(i, j,k,l, m, num_of_fermions)
        vertex = get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,false,true)*get_JW_matrix(l, num_of_fermions,false,true)*get_JW_matrix(m, num_of_fermions,false,true)

        for b in 1:2^5-1
            if count_ones(b) % 2 == 0
                binary =Bool.(digits(b, base=2, pad=5))
                vertex += get_JW_matrix(i, num_of_fermions,binary[1],true)*get_JW_matrix(j, num_of_fermions,binary[2],true)*get_JW_matrix(k, num_of_fermions,binary[3],true)*get_JW_matrix(l, num_of_fermions,binary[4],true)*get_JW_matrix(m, num_of_fermions,binary[5],true)
            end
        end
        return vertex
    end
    # vertex_tensor_5(i, j,k,l, m, num_of_fermions) = get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,false,true)*get_JW_matrix(l, num_of_fermions,false,true)*get_JW_matrix(m, num_of_fermions,false,true) + 
    #                                         get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,false,true)*get_JW_matrix(l, num_of_fermions,false,true)*get_JW_matrix(m, num_of_fermions,false,true) + 
    #                                         get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,true,true)*get_JW_matrix(l, num_of_fermions,false,true)*get_JW_matrix(m, num_of_fermions,false,true) + 
    #                                         get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,true,true)*get_JW_matrix(l, num_of_fermions,false,true)*get_JW_matrix(m, num_of_fermions,false,true) +
    #                                         get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,false,true)*get_JW_matrix(l, num_of_fermions,true,true)*get_JW_matrix(m, num_of_fermions,false,true) +
    #                                         get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,false,true)*get_JW_matrix(l, num_of_fermions,false,true)*get_JW_matrix(m, num_of_fermions,true,true) +
    #                                         get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,false,true)*get_JW_matrix(l, num_of_fermions,true,true)*get_JW_matrix(m, num_of_fermions,false,true) +
    #                                         get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,false,true)*get_JW_matrix(l, num_of_fermions,false,true)*get_JW_matrix(m, num_of_fermions,true,true) +
    #                                         get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,true,true)*get_JW_matrix(l, num_of_fermions,true,true)*get_JW_matrix(m, num_of_fermions,false,true) +
    #                                         get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,true,true)*get_JW_matrix(l, num_of_fermions,false,true)*get_JW_matrix(m, num_of_fermions,true,true) +
    #                                         get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,false,true)*get_JW_matrix(l, num_of_fermions,true,true)*get_JW_matrix(m, num_of_fermions,true,true) +
    #                                         get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,true,true)*get_JW_matrix(l, num_of_fermions,true,true)*get_JW_matrix(m, num_of_fermions,true,true) +
    #                                         get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,false,true)*get_JW_matrix(k, num_of_fermions,true,true)*get_JW_matrix(l, num_of_fermions,true,true)*get_JW_matrix(m, num_of_fermions,true,true) +
    #                                         get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,false,true)*get_JW_matrix(l, num_of_fermions,true,true)*get_JW_matrix(m, num_of_fermions,true,true) +
    #                                         get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,true,true)*get_JW_matrix(l, num_of_fermions,false,true)*get_JW_matrix(m, num_of_fermions,true,true) +
    #                                         get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(j, num_of_fermions,true,true)*get_JW_matrix(k, num_of_fermions,true,true)*get_JW_matrix(l, num_of_fermions,true,true)*get_JW_matrix(m, num_of_fermions,false,true);
    # Should be possible to simplify vertex tensor to just a single binary column with a 1 everywhere it should map e.g. for 3 vertex 000:1, 110: 1, 101: 1, 011:1 therefore column will be 10010110   
    # vertex_tensor_5(i,j,k,l,m, num_of_fermions) = [count(c -> c == '1', bitstring(i)[i,j,k,l,m]) % 2 == 0 for i in 1:2^num_of_fermions]


    function GHZ_tensor(j,k, num_of_fermions)
        spin_1 = SparseArray{Int8}([1; 0])
        spin_0 = SparseArray{Int8}([0; 1])
        @tensor GHZ_tensor[a,b,c]  := spin_1[a]*(get_JW_matrix(j, num_of_fermions,false, false)*get_JW_matrix(k, num_of_fermions,false,false))[b,c] +
        spin_0[a]*(get_JW_matrix(j, num_of_fermions,true,false)*get_JW_matrix(k, num_of_fermions,true,false))[b,c]
        return GHZ_tensor
    end

    X_tensor(i,num_of_fermions, on) = on ? get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(i, num_of_fermions,true,false) + get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(i, num_of_fermions,false,false) : SparseArray{Int8}(sparse(I(2^(num_of_fermions))))
    Z_tensor(i,num_of_fermions, on) = on ? get_JW_matrix(i, num_of_fermions,false,true)*get_JW_matrix(i, num_of_fermions,false,false) - get_JW_matrix(i, num_of_fermions,true,true)*get_JW_matrix(i, num_of_fermions,true,false) : SparseArray{Int8}(sparse(I(2^(num_of_fermions))))


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

    function multiply_by_vacuum(MPO, num_of_fermions, num_of_physical_fermions)
        partial_vaccum = SparseArray(sparse(reshape([j==((k-1)*2^(num_of_fermions-num_of_physical_fermions))+1 ? 1 : 0 for j in 1:2^(num_of_fermions) for k in 1:2^num_of_physical_fermions],2^num_of_physical_fermions,2^num_of_fermions)))
        full_vacuum = SparseArray(sparse([j == 1 ? 1 : 0 for j in 1:2^num_of_fermions]))
        @tensor MPO_without_virtual_fermions[b,a] := partial_vaccum[b,e]*MPO[a,e,f]*full_vacuum[f] # Finally correct!
        return MPO_without_virtual_fermions
    end
    
    function get_1D_KW(num_of_sites,X_defect, Z_defect)
        CNOT = [1; 0]⊗[1; 0]⊗[1;  0]+ [0 ; 1]⊗[0 ; 1]⊗[1;  0] + [0; 1]⊗[1 ; 0]⊗[0 ; 1] + [1;  0]⊗[0; 1]⊗[0 ; 1]
        GHZ = [1; 0]⊗[1; 0]⊗[1;  0]+[0; 1]⊗[0; 1]⊗[0;  1]
        @tensor MPO[a,b,c,e] := CNOT[a,b,d]*GHZ[d,c,e]
        for i in 2:num_of_sites # 2.5
            @tensor MPO[a,b,f,c,i,e] := MPO[a,b,c,g]*CNOT[g,f,h]*GHZ[h, i, e]
            MPO = reshape(MPO, 2, 2^i, 2^i, 2)
        end
        if X_defect
            @tensor MPO[a,b,c,e] := MPO[a,b,c,f]*[0 1; 1 0][f,e]
        end
        if Z_defect
            @tensor MPO[a,b,c,e] := MPO[a,b,c,f]*[1 0; 0 -1][f,e]
        end

        @tensor MPO[b,c] := MPO[a,b,c,a]
        reshape(MPO,2^num_of_sites, 2^num_of_sites)
    end

    get_1D_KW(4,false, false)

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

        traced_MPO = multiply_by_vacuum(trace_out_fermion(MPO,[j for j in 1:2*num_of_sites],  3*num_of_sites), 3*num_of_sites, num_of_sites) # 6.5

        if shift != 0
            # Translate spins
            spins_tensor_product_dims = (2 for j in 1:num_of_sites) |> Tuple
            # println([j== 1 ? 1 : ((j+num_of_sites-shift-2) % num_of_sites)+2 for j in 1:num_of_sites+1] )
            traced_MPO = reshape(permutedims(reshape(traced_MPO, (2^(num_of_sites),spins_tensor_product_dims...)),
            [j== 1 ? 1 : ((j+num_of_sites-shift-2) % num_of_sites)+2 for j in 1:num_of_sites+1]),
            2^(num_of_sites), 2^(num_of_sites))
        end 
        return traced_MPO
    end

    function get_2D_PEPO(X_defect,Z_defect_V,Z_defect_H, horizontal_shift, vertical_shift)

        MPO = vertex_tensor_5(1,2,3,4,horizontal_shift & vertical_shift ? 20 : vertical_shift ? 19 : horizontal_shift ? 18 : 17,20)*
        vertex_tensor_5(5,6,7,8,horizontal_shift & vertical_shift ? 19 : vertical_shift ? 20 : horizontal_shift ? 17 : 18,20)*
        vertex_tensor_5(9,10,11,12,horizontal_shift & vertical_shift ? 18 : vertical_shift ? 17 : horizontal_shift ? 20 : 19,20)*
        vertex_tensor_5(13,14,15,16,horizontal_shift & vertical_shift ? 17 : vertical_shift ? 18 : horizontal_shift ? 19 : 20,20)
        
        @tensor MPO[c,a,d] := MPO[a,b]*GHZ_tensor(1, 7, 20)[c,b,d]
        MPO = reshape(MPO, 2^1,2^(20), 2^(20))
        @tensor MPO[a,c,b,d] := MPO[a,b,e]*GHZ_tensor(5, 3, 20)[c,e,d]
        MPO = reshape(MPO, 2^2,2^(20), 2^(20))
        @tensor MPO[a,c,b,d] := MPO[a,b,e]*GHZ_tensor(9, 15, 20)[c,e,d]
        MPO = reshape(MPO, 2^3,2^(20), 2^(20))
        @tensor MPO[a,c,b,d] := MPO[a,b,e]*GHZ_tensor(13, 11, 20)[c,e,d]
        MPO = reshape(MPO, 2^4,2^(20), 2^(20))
        @tensor MPO[a,c,b,d] := MPO[a,b,e]*GHZ_tensor(4, 10, 20)[c,e,d]
        MPO = reshape(MPO, 2^5,2^(20), 2^(20))
        @tensor MPO[a,c,b,d] := MPO[a,b,e]*GHZ_tensor(12, 2, 20)[c,e,d]
        MPO = reshape(MPO, 2^6,2^(20), 2^(20))
        @tensor MPO[a,c,b,d] := MPO[a,b,e]*GHZ_tensor(8, 14, 20)[c,e,d]
        MPO = reshape(MPO, 2^7,2^(20), 2^(20))
        @tensor MPO[a,c,b,d] := MPO[a,b,e]*GHZ_tensor(16, 6, 20)[c,e,d]
        MPO = reshape(MPO, 2^8,2^(20), 2^(20))
        MPO = reshape(MPO, 2^(28), 2^(20))
        MPO = MPO*Z_tensor(5,20,Z_defect_V)*Z_tensor(13,20,Z_defect_V)*Z_tensor(12,20,Z_defect_H)*Z_tensor(16,20,Z_defect_H)*X_tensor(7,20,X_defect)
        MPO = reshape(MPO, 2^8,2^(20), 2^(20))

        traced_MPO = multiply_by_vacuum(trace_out_fermion(MPO,[j for j in 1:16],  20), 20, 4)

        if horizontal_shift
            traced_MPO = reshape(permutedims(reshape(traced_MPO, 16,2,2,2,2,2,2,2,2),[1,3,2,5,4,8,9,6,7]), 16, 256)
        end

        if vertical_shift
            traced_MPO = reshape(permutedims(reshape(traced_MPO, 16,2,2,2,2,2,2,2,2),[1,4,5,2,3,7,6,9,8]), 16, 256)
        end
        return traced_MPO
    end


    function get_1D_torus(X_defect,Z_defect)

        MPO = vertex_tensor_5(2,7,1,8,13 ,15)*vertex_tensor_5(4,9,3,10,14 ,15)*vertex_tensor_5(6,11,5,12,15,15)        
        
        @tensor MPO[c,a,d] := MPO[a,b]*GHZ_tensor(6, 1, 15)[c,b,d]
        MPO = reshape(MPO, 2^1,2^(15), 2^(15))
        @tensor MPO[a,c,b,d] := MPO[a,b,e]*GHZ_tensor(2, 3, 15)[c,e,d]
        MPO = reshape(MPO, 2^2,2^(15), 2^(15))
        @tensor MPO[a,c,b,d] := MPO[a,b,e]*GHZ_tensor(4, 5, 15)[c,e,d]
        MPO = reshape(MPO, 2^3,2^(15), 2^(15))
        @tensor MPO[a,c,b,d] := MPO[a,b,e]*GHZ_tensor(7, 8, 15)[c,e,d]
        MPO = reshape(MPO, 2^4,2^(15), 2^(15))
        @tensor MPO[a,c,b,d] := MPO[a,b,e]*GHZ_tensor(9, 10, 15)[c,e,d]
        MPO = reshape(MPO, 2^5,2^(15), 2^(15))
        @tensor MPO[a,c,b,d] := MPO[a,b,e]*GHZ_tensor(11, 12, 15)[c,e,d]

        MPO = reshape(MPO, 2^6,2^(15), 2^(15))
        MPO = reshape(MPO, 2^(21), 2^(15))
        MPO = MPO*Z_tensor(9,15,Z_defect)*X_tensor(9,15,X_defect)
        MPO = reshape(MPO, 2^6,2^(15), 2^(15))

        traced_MPO = multiply_by_vacuum(trace_out_fermion(MPO,[j for j in 1:12],  15), 15, 3)

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

    function get_1D_table(num_of_sites)
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
    end

function get_2D_table()
    for Z_H in (false, true)
        for Z_V in (false, true)
            for X_defect in (false, true)
                pepo = get_2D_PEPO(X_defect,Z_V,Z_H, false,false)
                pepo_twisted_horizontal = get_2D_PEPO(X_defect, Z_V, Z_H, true,false)
                pepo_twisted_vertical = get_2D_PEPO(X_defect, Z_V, Z_H, false,true)

                # Matrix(pepo)
                println("Defect: ", Z_H & Z_V & X_defect ? "Z_HZ_VX" : Z_H & Z_V ? "Z_HZ_V" : Z_H & X_defect ? "Z_HX" :  Z_V & X_defect ? "Z_VX" :  Z_H  ? "Z_H" :  Z_V ? "Z_V" :   X_defect ? "X" : "I")
                reshape(permutedims(Z ⊗ Z ⊗ Z ⊗ Z,[1,3,5,7,2,4,6,8]),16,16)*pepo == pepo ? println("ZZZZ: +1") : println("ZZZZ: -1")
                pepo*reshape(permutedims(X ⊗ X ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ Z ⊗ I(2) ⊗ Z ,[1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16]),256,256)
                pepo*reshape(permutedims(X ⊗ X ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ Z ⊗ I(2) ⊗ Z ,[1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16]),256,256) == pepo
                pepo*reshape(permutedims(X ⊗ X ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ Z ⊗ I(2) ⊗ Z ,[1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16]),256,256) == pepo ? println("XXIIIZIZ: +1") : println("XXIIIZIZ: -1")
                pepo*reshape(permutedims(I(2) ⊗ I(2) ⊗ X ⊗ X ⊗ Z ⊗ I(2) ⊗ Z ⊗ I(2) ,[1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16]),256,256) == pepo ? println("IIXXZIZI: +1") : println("IIXXZIZI: -1")
                pepo*reshape(permutedims(Z ⊗ I(2) ⊗ Z ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ X ⊗ X ,[1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16]),256,256) == pepo ? println("ZIZIIIXX: +1") : println("ZIZIIIXX: -1")
                pepo*reshape(permutedims(I(2) ⊗ Z ⊗I(2) ⊗ Z ⊗ X ⊗ X ⊗ I(2) ⊗ I(2) ,[1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16]),256,256) == pepo ? println("IZIZXXII: +1") : println("IZIZXXII: -1")
                pepo_twisted_horizontal == pepo ? println("Horizontal Twist: I") : print("")
                reshape(permutedims(Z ⊗ I(2) ⊗ Z ⊗ I(2),[1,3,5,7,2,4,6,8]),16,16)*pepo_twisted_horizontal == pepo ? println("Horizontal Twist: Z_V") : print("")
                pepo_twisted_horizontal*reshape(permutedims(I(2) ⊗ X ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ Z ,[1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16]),256,256) == pepo ? println("Horizontal Twist: IXIIIIIZ") : print("")
                reshape(permutedims(Z ⊗ I(2) ⊗ Z ⊗ I(2),[1,3,5,7,2,4,6,8]),16,16)*pepo_twisted_horizontal*reshape(permutedims(I(2) ⊗ X ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ Z ,[1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16]),256,256) == pepo ? println("Horizontal Twist: Z_VIXIIIIIZ") : print("")

                pepo_twisted_vertical == pepo ? println("Vertical Twist: I") : print("")
                reshape(permutedims(Z ⊗ Z ⊗ I(2) ⊗ I(2),[1,3,5,7,2,4,6,8]),16,16)*pepo_twisted_vertical == pepo ? println("Vertical Twist: Z_H") : print("")
                pepo_twisted_vertical*reshape(permutedims(Z ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ -X ⊗ I(2) ,[1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16]),256,256)==Matrix(pepo) ? println("Vertical Twist: ZIIIIIXI") : print("")
                reshape(permutedims(Z ⊗ Z ⊗ I(2) ⊗ I(2),[1,3,5,7,2,4,6,8]),16,16)*pepo_twisted_vertical*reshape(permutedims(Z ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ I(2) ⊗ -X ⊗ I(2) ,[1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16]),256,256)==Matrix(pepo) ? println("Vertical Twist: Z_HZIIIIIXI") : print("")
                
                gauge_condition = sparse(reshape(permutedims((Z*X) ⊗ Z ⊗ X ⊗ I(2) ⊗ (Z*X) ⊗ Z ⊗ X ⊗ I(2) ,[1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16]),256,256))
                pepo * gauge_condition == pepo ? println("Gauge condition 1 valid") : println("Gauge condition 1 invalid")

                gauge_condition_2 = sparse(reshape(permutedims(I(2)⊗ X ⊗ Z ⊗ (Z*X) ⊗ I(2) ⊗ X ⊗ Z ⊗ (Z*X) ,[1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16]),256,256))
                pepo * gauge_condition_2 == pepo ? println("Gauge condition 2 valid") : println("Gauge condition 2 invalid")
            end
        end
    end
end

function check_unitary()
    sum = sparse(zeros(256,256))
    for Z_H in (false, true)
        for Z_V in (false, true)
            for X_defect in (false,true)
                pepo = sparse(get_2D_PEPO(X_defect,Z_V,Z_H, false, false))
                sum += pepo'*pepo
                display(sum)
            end
        end
    end

    gauge_condition = sparse(reshape(permutedims((Z*X) ⊗ Z ⊗ X ⊗ I(2) ⊗ (Z*X) ⊗ Z ⊗ X ⊗ I(2) ,[1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16]),256,256))
    gauge_condition_2 = sparse(reshape(permutedims(I(2)⊗ X ⊗ Z ⊗ (Z*X) ⊗ I(2) ⊗ X ⊗ Z ⊗ (Z*X) ,[1,3,5,7,9,11,13,15,2,4,6,8,10,12,14,16]),256,256))

    sum - 8*gauge_condition - 8*gauge_condition_2 - 8*gauge_condition*gauge_condition_2 == 8*I(256)

    sum = sparse(zeros(16,16))
    for Z_H in (false, true)
        for Z_V in (false, true)
            for X_defect in (false,true)
                pepo = sparse(get_2D_PEPO(X_defect,Z_V,Z_H, false, false))
                sum += pepo*pepo'
                display(sum)
            end
        end
    end
    sum == 8*I(16)
end

# Check 1D torus is the same as cycle
reshape(reshape(Matrix(get_1D_torus(false,false)),2^3,2,2,2,2,2,2)[:,:,:,:,1,1,1],8,8) == Matrix(get_1D_MPO(3,false,false,0)) ? println("1D torus with |0> on extra spins matches 1D cycle") : println("Error")


num_of_sites = 2
get_1D_KW(num_of_sites,true, false)
# Matrix(get_1D_MPO(num_of_sites,false,false,0))*(get_1D_KW(num_of_sites,false,false)+get_1D_KW(num_of_sites,false, true))'
# Matrix(get_1D_MPO(num_of_sites,false,false,0))*(get_1D_KW(num_of_sites,true,false)+get_1D_KW(num_of_sites,true, true))'

Matrix(get_1D_MPO(num_of_sites,false,false,0))*get_1D_KW(num_of_sites,false, true)'
Matrix(get_1D_MPO(num_of_sites,false,true,0))*get_1D_KW(num_of_sites,false, false)'
Matrix(get_1D_MPO(num_of_sites,true,false,0))*get_1D_KW(num_of_sites,true, false)'
Matrix(get_1D_MPO(num_of_sites,true,true,0))*get_1D_KW(num_of_sites,true, true)'