module {
  func.func @main(%arg0: tensor<91x49x94xi32>, %arg1: tensor<1x49x94xi32>, %arg2: tensor<5x40x91xi1>, %arg3: tensor<f32>) -> (tensor<91x49x94xi32>, tensor<5x40x91xi1>, tensor<f32>) {
    %0 = tosa.maximum %arg0, %arg1 : (tensor<91x49x94xi32>, tensor<1x49x94xi32>) -> tensor<91x49x94xi32>
    %1 = tosa.logical_not %arg2 : (tensor<5x40x91xi1>) -> tensor<5x40x91xi1>
    %2 = tosa.reciprocal %arg3 : (tensor<f32>) -> tensor<f32>
    return %0, %1, %2 : tensor<91x49x94xi32>, tensor<5x40x91xi1>, tensor<f32>
  }
}
