module {
  func.func @main(%arg0: tensor<i8>, %arg1: tensor<20x72x29x97xi8>, %arg2: tensor<f32>) -> (tensor<20x72x1x97xi8>, tensor<f32>, tensor<i1>) {
    %0 = tosa.abs %arg0 : (tensor<i8>) -> tensor<i8>
    %1 = tosa.greater_equal %0, %0 : (tensor<i8>, tensor<i8>) -> tensor<i1>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %3 = tosa.reduce_product %arg1 {axis = 2 : i32} : (tensor<20x72x29x97xi8>) -> tensor<20x72x1x97xi8>
    %4 = tosa.exp %arg2 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.logical_right_shift %2, %2 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    return %3, %4, %5 : tensor<20x72x1x97xi8>, tensor<f32>, tensor<i1>
  }
}
