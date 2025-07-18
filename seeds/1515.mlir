module {
  func.func @main(%arg0: tensor<21x96xi1>, %arg1: tensor<21x1xi1>) -> tensor<3x12xi1> {
    %0 = tosa.logical_left_shift %arg0, %arg1 : (tensor<21x96xi1>, tensor<21x1xi1>) -> tensor<21x96xi1>
    %1 = tosa.reduce_product %0 {axis = 1 : i32} : (tensor<21x96xi1>) -> tensor<21x1xi1>
    %2 = tosa.bitwise_and %1, %1 : (tensor<21x1xi1>, tensor<21x1xi1>) -> tensor<21x1xi1>
    %3 = tosa.logical_or %2, %1 : (tensor<21x1xi1>, tensor<21x1xi1>) -> tensor<21x1xi1>
    %s_4_start = tosa.const_shape {values = dense<[ 3, 0 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_4_size = tosa.const_shape {values = dense<[ 3, 12 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %4 = tosa.slice %3, %s_4_start, %s_4_size : (tensor<21x1xi1>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<3x12xi1>
    return %4 : tensor<3x12xi1>
  }
}
