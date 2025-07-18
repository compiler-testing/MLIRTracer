module {
  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<29xi1>) -> (tensor<16xf32>, tensor<1xi1>) {
    %0 = tosa.log %arg0 : (tensor<8xf32>) -> tensor<8xf32>
    %1 = tosa.log %0 : (tensor<8xf32>) -> tensor<8xf32>
    %2 = tosa.abs %1 : (tensor<8xf32>) -> tensor<8xf32>
    %t_3 = tosa.const_shape {values = dense<[ 2 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.tile %2, %t_3 : (tensor<8xf32>, !tosa.shape<1>) -> tensor<16xf32>
    %4 = tosa.identity %3 : (tensor<16xf32>) -> tensor<16xf32>
    %5 = tosa.reverse %4 {axis = 0 : i32} : (tensor<16xf32>) -> tensor<16xf32>
    %6 = tosa.reverse %5 {axis = 0 : i32} : (tensor<16xf32>) -> tensor<16xf32>
    %7 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<29xi1>) -> tensor<1xi1>
    %8 = tosa.bitwise_or %7, %7 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %9 = tosa.sub %7, %7 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %10 = tosa.logical_right_shift %9, %8 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    return %6, %10 : tensor<16xf32>, tensor<1xi1>
  }
}
