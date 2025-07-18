module {
  func.func @main(%arg0: tensor<52xf32>, %arg1: tensor<1x2xi64>, %arg2: tensor<33xi1>) -> (tensor<1xi1>, tensor<52xi1>) {
    %p_0 = tosa.const_shape {values = dense<0> : tensor<2xindex>} : () -> !tosa.shape<2>
    %pad_const_0 = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %0 = tosa.pad %arg0, %p_0, %pad_const_0 : (tensor<52xf32>, !tosa.shape<2>, tensor<1xf32>) -> tensor<52xf32>
    %1 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<33xi1>) -> tensor<1xi1>
    %2 = tosa.reverse %0 {axis = 0 : i32} : (tensor<52xf32>) -> tensor<52xf32>
    %3 = tosa.minimum %2, %0 : (tensor<52xf32>, tensor<52xf32>) -> tensor<52xf32>
    %4 = tosa.greater %3, %2 : (tensor<52xf32>, tensor<52xf32>) -> tensor<52xi1>
    return %1, %4 : tensor<1xi1>, tensor<52xi1>
  }
}
