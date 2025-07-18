module {
  func.func @main(%arg0: tensor<20xi32>, %arg1: tensor<20xi32>, %arg2: tensor<73x24x76x14x69x8xf32>) -> (tensor<20xi1>, tensor<32x32156208xf32>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi32>
    %1 = tosa.bitwise_or %0, %0 : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi32>
    %2 = tosa.identity %1 : (tensor<20xi32>) -> tensor<20xi32>
    %3 = tosa.ceil %arg2 : (tensor<73x24x76x14x69x8xf32>) -> tensor<73x24x76x14x69x8xf32>
    %4 = tosa.greater %2, %1 : (tensor<20xi32>, tensor<20xi32>) -> tensor<20xi1>
    %r_5 = tosa.const_shape {values = dense<[ 32, 32156208 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %5 = tosa.reshape %3, %r_5 : (tensor<73x24x76x14x69x8xf32>, !tosa.shape<2>) -> tensor<32x32156208xf32>
    return %4, %5 : tensor<20xi1>, tensor<32x32156208xf32>
  }
}
