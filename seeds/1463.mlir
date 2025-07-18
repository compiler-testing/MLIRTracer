module {
  func.func @main(%arg0: tensor<i64>, %arg1: tensor<84x23x70x52xf32>) -> (tensor<84x23x70x52xf32>, tensor<1x1x1xi64>) {
    %r_0 = tosa.const_shape {values = dense<[ 1, 1 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %0 = tosa.reshape %arg0, %r_0 : (tensor<i64>, !tosa.shape<2>) -> tensor<1x1xi64>
    %r_1 = tosa.const_shape {values = dense<[ 1, 1, 1 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %1 = tosa.reshape %0, %r_1 : (tensor<1x1xi64>, !tosa.shape<3>) -> tensor<1x1x1xi64>
    %2 = tosa.log %arg1 : (tensor<84x23x70x52xf32>) -> tensor<84x23x70x52xf32>
    %3 = tosa.rsqrt %2 : (tensor<84x23x70x52xf32>) -> tensor<84x23x70x52xf32>
    %4 = tosa.bitwise_and %1, %1 : (tensor<1x1x1xi64>, tensor<1x1x1xi64>) -> tensor<1x1x1xi64>
    return %3, %4 : tensor<84x23x70x52xf32>, tensor<1x1x1xi64>
  }
}
