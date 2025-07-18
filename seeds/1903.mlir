module {
  func.func @main(%arg0: tensor<91x58x85x93x1xi8>, %arg1: tensor<1x1x85x93x1xi8>, %arg2: tensor<83x6x14x70x8xf32>, %arg3: tensor<40x94x86xi8>) -> (tensor<1x7x4x8x8xi1>, tensor<6x1x9x2x9xf32>, tensor<6x1x9x2x9xf32>, tensor<40x94x1xi8>) {
    %0 = tosa.greater_equal %arg0, %arg1 : (tensor<91x58x85x93x1xi8>, tensor<1x1x85x93x1xi8>) -> tensor<91x58x85x93x1xi1>
    %1 = tosa.floor %arg2 : (tensor<83x6x14x70x8xf32>) -> tensor<83x6x14x70x8xf32>
    %2 = tosa.clz %0 : (tensor<91x58x85x93x1xi1>) -> tensor<91x58x85x93x1xi1>
    %s_3_start = tosa.const_shape {values = dense<[ 56, 51, 40, 57, 0 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_3_size = tosa.const_shape {values = dense<[ 1, 7, 4, 8, 8 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %3 = tosa.slice %2, %s_3_start, %s_3_size : (tensor<91x58x85x93x1xi1>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<1x7x4x8x8xi1>
    %s_4_start = tosa.const_shape {values = dense<[ 50, 5, 5, 1, 0 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %s_4_size = tosa.const_shape {values = dense<[ 6, 1, 9, 2, 9 ]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %4 = tosa.slice %1, %s_4_start, %s_4_size : (tensor<83x6x14x70x8xf32>, !tosa.shape<5>, !tosa.shape<5>) -> tensor<6x1x9x2x9xf32>
    %5 = tosa.tanh %4 : (tensor<6x1x9x2x9xf32>) -> tensor<6x1x9x2x9xf32>
    %in_zp_6 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %out_zp_6 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %6 = tosa.negate %4, %in_zp_6, %out_zp_6 : (tensor<6x1x9x2x9xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<6x1x9x2x9xf32>
    %7 = tosa.reduce_min %arg3 {axis = 2 : i32} : (tensor<40x94x86xi8>) -> tensor<40x94x1xi8>
    return %3, %5, %6, %7 : tensor<1x7x4x8x8xi1>, tensor<6x1x9x2x9xf32>, tensor<6x1x9x2x9xf32>, tensor<40x94x1xi8>
  }
}
