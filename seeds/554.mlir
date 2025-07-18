module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<9xi8>) -> (tensor<f32>, tensor<1xi8>) {
    %0 = tosa.sub %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %1 = tosa.reduce_product %arg2 {axis = 0 : i32} : (tensor<9xi8>) -> tensor<1xi8>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<1xi8>, tensor<1xi8>) -> tensor<1xi8>
    return %0, %2 : tensor<f32>, tensor<1xi8>
  }
}
