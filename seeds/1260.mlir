module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<90x58x11x31xi8>) -> (tensor<f32>, tensor<90x58x1x31xi8>) {
    %0 = tosa.rsqrt %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.reduce_product %arg1 {axis = 2 : i32} : (tensor<90x58x11x31xi8>) -> tensor<90x58x1x31xi8>
    return %0, %1 : tensor<f32>, tensor<90x58x1x31xi8>
  }
}
