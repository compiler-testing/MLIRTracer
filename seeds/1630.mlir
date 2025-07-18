module {
  func.func @main(%arg0: tensor<91x21x7x33x18xi16>, %arg1: tensor<4x27x94x74x78x32xf32>) -> (tensor<1x174636xi16>, tensor<1x174636xi16>, tensor<4x27x94x74x78x32xi1>) {
    %r_0 = tosa.const_shape {values = dense<[ 91, 87318 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<91x21x7x33x18xi16>, !tosa.shape<2>) -> tensor<91x87318xi16>
    %t_1 = tosa.const_shape {values = dense<[ 1, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %1 = tosa.tile %0, %t_1 : (tensor<91x87318xi16>, !tosa.shape<2>) -> tensor<91x174636xi16>
    %2 = tosa.reduce_product %1 {axis = 0 : i32} : (tensor<91x174636xi16>) -> tensor<1x174636xi16>
    %3 = tosa.exp %arg1 : (tensor<4x27x94x74x78x32xf32>) -> tensor<4x27x94x74x78x32xf32>
    %4 = tosa.logical_left_shift %2, %2 : (tensor<1x174636xi16>, tensor<1x174636xi16>) -> tensor<1x174636xi16>
    %5 = tosa.greater_equal %3, %3 : (tensor<4x27x94x74x78x32xf32>, tensor<4x27x94x74x78x32xf32>) -> tensor<4x27x94x74x78x32xi1>
    %6 = tosa.reduce_min %1 {axis = 0 : i32} : (tensor<91x174636xi16>) -> tensor<1x174636xi16>
    %7 = tosa.bitwise_xor %5, %5 : (tensor<4x27x94x74x78x32xi1>, tensor<4x27x94x74x78x32xi1>) -> tensor<4x27x94x74x78x32xi1>
    %8 = tosa.logical_right_shift %7, %7 : (tensor<4x27x94x74x78x32xi1>, tensor<4x27x94x74x78x32xi1>) -> tensor<4x27x94x74x78x32xi1>
    %9 = tosa.bitwise_not %8 : (tensor<4x27x94x74x78x32xi1>) -> tensor<4x27x94x74x78x32xi1>
    return %4, %6, %9 : tensor<1x174636xi16>, tensor<1x174636xi16>, tensor<4x27x94x74x78x32xi1>
  }
}
