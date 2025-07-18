module {
  func.func @main(%arg0: tensor<25x99x86x82xf32>, %arg1: tensor<69x54x2x38x13x88xi32>, %arg2: tensor<55x53xi1>) -> (tensor<25x99x86x82xf32>, tensor<69x54x2x38x13x88xi32>, tensor<1x53xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<25x99x86x82xf32>) -> tensor<25x99x86x82xf32>
    %1 = tosa.clz %arg1 : (tensor<69x54x2x38x13x88xi32>) -> tensor<69x54x2x38x13x88xi32>
    %2 = tosa.clz %1 : (tensor<69x54x2x38x13x88xi32>) -> tensor<69x54x2x38x13x88xi32>
    %3 = tosa.reduce_any %arg2 {axis = 0 : i32} : (tensor<55x53xi1>) -> tensor<1x53xi1>
    return %0, %2, %3 : tensor<25x99x86x82xf32>, tensor<69x54x2x38x13x88xi32>, tensor<1x53xi1>
  }
}
