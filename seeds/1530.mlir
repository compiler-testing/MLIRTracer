module {
  func.func @main(%arg0: tensor<24x19x23xi64>, %arg1: tensor<1x19x23xi64>, %arg2: tensor<59x89x64x2xi1>, %arg3: tensor<82xf32>) -> (tensor<228x1x1x46xi64>, tensor<336064xi1>, tensor<1xf32>, tensor<118x178x64x1xi1>) {
    %0 = tosa.logical_right_shift %arg0, %arg1 : (tensor<24x19x23xi64>, tensor<1x19x23xi64>) -> tensor<24x19x23xi64>
    %r_1 = tosa.const_shape {values = dense<[ 228, 1, 1, 46 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.reshape %0, %r_1 : (tensor<24x19x23xi64>, !tosa.shape<4>) -> tensor<228x1x1x46xi64>
    %2 = tosa.reduce_all %arg2 {axis = 3 : i32} : (tensor<59x89x64x2xi1>) -> tensor<59x89x64x1xi1>
    %3 = tosa.tanh %arg3 : (tensor<82xf32>) -> tensor<82xf32>
    %4 = tosa.bitwise_xor %1, %1 : (tensor<228x1x1x46xi64>, tensor<228x1x1x46xi64>) -> tensor<228x1x1x46xi64>
    %5 = tosa.logical_not %2 : (tensor<59x89x64x1xi1>) -> tensor<59x89x64x1xi1>
    %6 = tosa.arithmetic_right_shift %5, %5 {round = false} : (tensor<59x89x64x1xi1>, tensor<59x89x64x1xi1>) -> tensor<59x89x64x1xi1>
    %7 = tosa.floor %3 : (tensor<82xf32>) -> tensor<82xf32>
    %r_8 = tosa.const_shape {values = dense<[ 336064 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %8 = tosa.reshape %2, %r_8 : (tensor<59x89x64x1xi1>, !tosa.shape<1>) -> tensor<336064xi1>
    %9 = tosa.arithmetic_right_shift %2, %6 {round = false} : (tensor<59x89x64x1xi1>, tensor<59x89x64x1xi1>) -> tensor<59x89x64x1xi1>
    %t_10 = tosa.const_shape {values = dense<[ 2, 2, 1, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %10 = tosa.tile %9, %t_10 : (tensor<59x89x64x1xi1>, !tosa.shape<4>) -> tensor<118x178x64x1xi1>
    %11 = tosa.reduce_min %7 {axis = 0 : i32} : (tensor<82xf32>) -> tensor<1xf32>
    %12 = tosa.pow %11, %11 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %13 = tosa.minimum %12, %12 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %14 = tosa.log %13 : (tensor<1xf32>) -> tensor<1xf32>
    %15 = tosa.arithmetic_right_shift %10, %10 {round = false} : (tensor<118x178x64x1xi1>, tensor<118x178x64x1xi1>) -> tensor<118x178x64x1xi1>
    return %4, %8, %14, %15 : tensor<228x1x1x46xi64>, tensor<336064xi1>, tensor<1xf32>, tensor<118x178x64x1xi1>
  }
}
