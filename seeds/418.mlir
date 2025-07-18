module {
  func.func @main(%arg0: tensor<80x16xi32>, %arg1: tensor<1x1xi32>, %arg2: tensor<38x23x19xf32>) -> (tensor<1x5xi1>, tensor<1x23x1xf32>, tensor<38x1xi32>) {
    %0 = tosa.bitwise_and %arg0, %arg1 : (tensor<80x16xi32>, tensor<1x1xi32>) -> tensor<80x16xi32>
    %1 = tosa.intdiv %0, %0 : (tensor<80x16xi32>, tensor<80x16xi32>) -> tensor<80x16xi32>
    %2 = tosa.reduce_sum %1 {axis = 0 : i32} : (tensor<80x16xi32>) -> tensor<1x16xi32>
    %s_3_start = tosa.const_shape {values = dense<[ 0, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %s_3_size = tosa.const_shape {values = dense<[ 8, 5 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.slice %2, %s_3_start, %s_3_size : (tensor<1x16xi32>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<8x5xi32>
    %4 = tosa.equal %3, %3 : (tensor<8x5xi32>, tensor<8x5xi32>) -> tensor<8x5xi1>
    %5 = tosa.reduce_any %4 {axis = 0 : i32} : (tensor<8x5xi1>) -> tensor<1x5xi1>
    %6 = tosa.sigmoid %arg2 : (tensor<38x23x19xf32>) -> tensor<38x23x19xf32>
    %7 = tosa.reduce_min %6 {axis = 2 : i32} : (tensor<38x23x19xf32>) -> tensor<38x23x1xf32>
    %8 = tosa.reduce_min %5 {axis = 0 : i32} : (tensor<1x5xi1>) -> tensor<1x5xi1>
    %9 = tosa.minimum %7, %7 : (tensor<38x23x1xf32>, tensor<38x23x1xf32>) -> tensor<38x23x1xf32>
    %10 = tosa.reduce_max %7 {axis = 0 : i32} : (tensor<38x23x1xf32>) -> tensor<1x23x1xf32>
    %11 = tosa.add %10, %10 : (tensor<1x23x1xf32>, tensor<1x23x1xf32>) -> tensor<1x23x1xf32>
    %12 = tosa.argmax %9 {axis = 1 : i32} : (tensor<38x23x1xf32>) -> tensor<38x1xi32>
    return %8, %11, %12 : tensor<1x5xi1>, tensor<1x23x1xf32>, tensor<38x1xi32>
  }
}
