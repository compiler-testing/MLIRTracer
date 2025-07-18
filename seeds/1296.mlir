module {
  func.func @main(%arg0: tensor<61x69x30xf32>, %arg1: tensor<49x13x17x2xi1>) -> (tensor<61x69x30xf32>, tensor<49x13x1x2xi1>) {
    %0 = tosa.floor %arg0 : (tensor<61x69x30xf32>) -> tensor<61x69x30xf32>
    %1 = tosa.reduce_all %arg1 {axis = 2 : i32} : (tensor<49x13x17x2xi1>) -> tensor<49x13x1x2xi1>
    return %0, %1 : tensor<61x69x30xf32>, tensor<49x13x1x2xi1>
  }
}
