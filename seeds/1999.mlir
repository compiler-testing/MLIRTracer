module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<29x58xi8>) -> (tensor<f32>, tensor<29x1xi8>) {
    %0 = tosa.sigmoid %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.rsqrt %0 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.tanh %1 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.reduce_product %arg1 {axis = 1 : i32} : (tensor<29x58xi8>) -> tensor<29x1xi8>
    return %2, %3 : tensor<f32>, tensor<29x1xi8>
  }
}
