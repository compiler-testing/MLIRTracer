module {
  func.func @main(%arg0: tensor<57xf32>, %arg1: tensor<17x85x18x57x15xi8>, %arg2: tensor<17x1x1x1x15xi8>, %arg3: tensor<i1>) -> (tensor<57xf32>, tensor<10x3x3x10x3xi8>, tensor<i1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<57xf32>) -> tensor<57xf32>
    %1 = tosa.bitwise_xor %arg1, %arg2 : (tensor<17x85x18x57x15xi8>, tensor<17x1x1x1x15xi8>) -> tensor<17x85x18x57x15xi8>
    %2 = tosa.sigmoid %0 : (tensor<57xf32>) -> tensor<57xf32>
    %3 = tosa.ceil %2 : (tensor<57xf32>) -> tensor<57xf32>
    %s_4_start = tosa.const_shape {values = dense<[ 1, 12, 12, 1, 4 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_4_size = tosa.const_shape {values = dense<[ 5, 3, 3, 10, 3 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %4 = tosa.slice %1, %s_4_start, %s_4_size : (tensor<17x85x18x57x15xi8>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<5x3x3x10x3xi8>
    %5 = tosa.pow %3, %2 : (tensor<57xf32>, tensor<57xf32>) -> tensor<57xf32>
    %6 = tosa.concat %4, %4 {axis = 0 : i32} : (tensor<5x3x3x10x3xi8>, tensor<5x3x3x10x3xi8>) -> tensor<10x3x3x10x3xi8>
    %7 = tosa.logical_not %arg3 : (tensor<i1>) -> tensor<i1>
    %in_zp_8 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_8 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %8 = tosa.negate %7, %in_zp_8, %out_zp_8 : (tensor<i1>, tensor<1xi1>, tensor<1xi1>) -> tensor<i1>
    return %5, %6, %8 : tensor<57xf32>, tensor<10x3x3x10x3xi8>, tensor<i1>
  }
}
