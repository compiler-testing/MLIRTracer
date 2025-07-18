module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<64x99x33x44xi32>) -> (tensor<f32>, tensor<64x99x33x1xi32>) {
    %0 = tosa.ceil %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.reduce_min %arg1 {axis = 3 : i32} : (tensor<64x99x33x44xi32>) -> tensor<64x99x33x1xi32>
    %2 = tosa.minimum %1, %1 : (tensor<64x99x33x1xi32>, tensor<64x99x33x1xi32>) -> tensor<64x99x33x1xi32>
    %3 = tosa.minimum %2, %2 : (tensor<64x99x33x1xi32>, tensor<64x99x33x1xi32>) -> tensor<64x99x33x1xi32>
    return %0, %3 : tensor<f32>, tensor<64x99x33x1xi32>
  }
}
