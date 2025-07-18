module {
  func.func @main(%arg0: tensor<31x73x10x38x72xf32>, %arg1: tensor<52x31x73xi64>, %arg2: tensor<49x41xi1>, %arg3: tensor<1x1xi1>) -> (tensor<31x73x10x38x72xf32>, tensor<52x1x73xi1>, tensor<49x41xi1>, tensor<1898x1x1x2xi64>) {
    %0 = tosa.ceil %arg0 : (tensor<31x73x10x38x72xf32>) -> tensor<31x73x10x38x72xf32>
    %1 = tosa.log %0 : (tensor<31x73x10x38x72xf32>) -> tensor<31x73x10x38x72xf32>
    %2 = tosa.reduce_max %arg1 {axis = 1 : i32} : (tensor<52x31x73xi64>) -> tensor<52x1x73xi64>
    %3 = tosa.logical_xor %arg2, %arg3 : (tensor<49x41xi1>, tensor<1x1xi1>) -> tensor<49x41xi1>
    %4 = tosa.bitwise_or %3, %3 : (tensor<49x41xi1>, tensor<49x41xi1>) -> tensor<49x41xi1>
    %r_5 = tosa.const_shape {values = dense<[ 1898, 1, 1, 2 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %5 = tosa.reshape %2, %r_5 : (tensor<52x1x73xi64>, !tosa.shape<4>) -> tensor<1898x1x1x2xi64>
    %6 = tosa.equal %2, %2 : (tensor<52x1x73xi64>, tensor<52x1x73xi64>) -> tensor<52x1x73xi1>
    %7 = tosa.logical_right_shift %4, %4 : (tensor<49x41xi1>, tensor<49x41xi1>) -> tensor<49x41xi1>
    %8 = tosa.maximum %5, %5 : (tensor<1898x1x1x2xi64>, tensor<1898x1x1x2xi64>) -> tensor<1898x1x1x2xi64>
    return %1, %6, %7, %8 : tensor<31x73x10x38x72xf32>, tensor<52x1x73xi1>, tensor<49x41xi1>, tensor<1898x1x1x2xi64>
  }
}
