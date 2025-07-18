module {
  func.func @main(%arg0: tensor<37x79x51xi32>, %arg1: tensor<1x79x1xi32>, %arg2: tensor<52x13xi1>, %arg3: tensor<52x1xi1>) -> (tensor<37x79x51xi32>, tensor<37x79x51xi1>, tensor<104x26xi1>, tensor<26xi32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<37x79x51xi32>, tensor<1x79x1xi32>) -> tensor<37x79x51xi32>
    %1 = tosa.reverse %0 {axis = 0 : i32} : (tensor<37x79x51xi32>) -> tensor<37x79x51xi32>
    %2 = tosa.logical_and %arg2, %arg3 : (tensor<52x13xi1>, tensor<52x1xi1>) -> tensor<52x13xi1>
    %t_3 = tosa.const_shape {values = dense<[ 2, 2 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.tile %2, %t_3 : (tensor<52x13xi1>, !tosa.shape<2>) -> tensor<104x26xi1>
    %4 = tosa.sub %1, %1 : (tensor<37x79x51xi32>, tensor<37x79x51xi32>) -> tensor<37x79x51xi32>
    %5 = tosa.bitwise_or %3, %3 : (tensor<104x26xi1>, tensor<104x26xi1>) -> tensor<104x26xi1>
    %6 = tosa.logical_and %5, %3 : (tensor<104x26xi1>, tensor<104x26xi1>) -> tensor<104x26xi1>
    %7 = tosa.greater_equal %1, %1 : (tensor<37x79x51xi32>, tensor<37x79x51xi32>) -> tensor<37x79x51xi1>
    %in_zp_8 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %out_zp_8 = "tosa.const"() <{values = dense<0> : tensor<1xi1>}> : () -> tensor<1xi1>
    %8 = tosa.negate %6, %in_zp_8, %out_zp_8 : (tensor<104x26xi1>, tensor<1xi1>, tensor<1xi1>) -> tensor<104x26xi1>
    %9 = tosa.reverse %5 {axis = 0 : i32} : (tensor<104x26xi1>) -> tensor<104x26xi1>
    %10 = tosa.clz %9 : (tensor<104x26xi1>) -> tensor<104x26xi1>
    %11 = tosa.argmax %8 {axis = 0 : i32} : (tensor<104x26xi1>) -> tensor<26xi32>
    return %4, %7, %10, %11 : tensor<37x79x51xi32>, tensor<37x79x51xi1>, tensor<104x26xi1>, tensor<26xi32>
  }
}
