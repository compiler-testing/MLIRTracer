module {
  func.func @main(%arg0: tensor<77xi32>, %arg1: tensor<77xi32>, %arg2: tensor<36xi1>, %arg3: tensor<16x24x74x74xf32>) -> (tensor<36xi1>, tensor<154xi32>, tensor<16x24x74x74xf32>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<77xi32>, tensor<77xi32>) -> tensor<77xi32>
    %1 = tosa.concat %0, %0 {axis = 0 : i32} : (tensor<77xi32>, tensor<77xi32>) -> tensor<154xi32>
    %2 = tosa.logical_not %arg2 : (tensor<36xi1>) -> tensor<36xi1>
    %3 = tosa.abs %1 : (tensor<154xi32>) -> tensor<154xi32>
    %4 = tosa.ceil %arg3 : (tensor<16x24x74x74xf32>) -> tensor<16x24x74x74xf32>
    %5 = tosa.log %4 : (tensor<16x24x74x74xf32>) -> tensor<16x24x74x74xf32>
    return %2, %3, %5 : tensor<36xi1>, tensor<154xi32>, tensor<16x24x74x74xf32>
  }
}
