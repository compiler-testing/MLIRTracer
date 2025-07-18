module {
  func.func @main(%arg0: tensor<13x69x61x12xi64>, %arg1: tensor<13x69x1x12xi64>, %arg2: tensor<f32>, %arg3: tensor<49x9x63x98xi1>, %arg4: tensor<49x1x63x1xi1>) -> (tensor<f32>, tensor<13x69x61x12xi64>, tensor<49x9x63x98xi1>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<13x69x61x12xi64>, tensor<13x69x1x12xi64>) -> tensor<13x69x61x12xi64>
    %1 = tosa.tanh %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.bitwise_xor %0, %0 : (tensor<13x69x61x12xi64>, tensor<13x69x61x12xi64>) -> tensor<13x69x61x12xi64>
    %3 = tosa.logical_or %arg3, %arg4 : (tensor<49x9x63x98xi1>, tensor<49x1x63x1xi1>) -> tensor<49x9x63x98xi1>
    return %1, %2, %3 : tensor<f32>, tensor<13x69x61x12xi64>, tensor<49x9x63x98xi1>
  }
}
