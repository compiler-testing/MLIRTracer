module {
  func.func @main(%arg0: tensor<17x30x37xi64>, %arg1: tensor<17x37x53xi64>, %arg2: tensor<11x37x33x50x84x82xf32>) -> (tensor<11x37x33x50x84x82xi1>, tensor<17x1x1xi64>, tensor<11x37x33x50x84x82xf32>) {
    %a_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %b_zp_0 = "tosa.const"() <{values = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %0 = tosa.matmul %arg0, %arg1, %a_zp_0, %b_zp_0 : (tensor<17x30x37xi64>, tensor<17x37x53xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<17x30x53xi64>
    %1 = tosa.reverse %0 {axis = 2 : i32} : (tensor<17x30x53xi64>) -> tensor<17x30x53xi64>
    %2 = tosa.floor %arg2 : (tensor<11x37x33x50x84x82xf32>) -> tensor<11x37x33x50x84x82xf32>
    %3 = tosa.reduce_min %1 {axis = 1 : i32} : (tensor<17x30x53xi64>) -> tensor<17x1x53xi64>
    %4 = tosa.bitwise_or %3, %3 : (tensor<17x1x53xi64>, tensor<17x1x53xi64>) -> tensor<17x1x53xi64>
    %5 = tosa.greater_equal %2, %2 : (tensor<11x37x33x50x84x82xf32>, tensor<11x37x33x50x84x82xf32>) -> tensor<11x37x33x50x84x82xi1>
    %6 = tosa.bitwise_and %4, %3 : (tensor<17x1x53xi64>, tensor<17x1x53xi64>) -> tensor<17x1x53xi64>
    %7 = tosa.reduce_min %6 {axis = 2 : i32} : (tensor<17x1x53xi64>) -> tensor<17x1x1xi64>
    %8 = tosa.pow %2, %2 : (tensor<11x37x33x50x84x82xf32>, tensor<11x37x33x50x84x82xf32>) -> tensor<11x37x33x50x84x82xf32>
    return %5, %7, %8 : tensor<11x37x33x50x84x82xi1>, tensor<17x1x1xi64>, tensor<11x37x33x50x84x82xf32>
  }
}
