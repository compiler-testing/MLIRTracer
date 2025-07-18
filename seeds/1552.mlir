module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<82x42xi1>) -> (tensor<f32>, tensor<1x1xi1>) {
    %0 = tosa.sigmoid %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.rsqrt %0 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<82x42xi1>) -> tensor<1x42xi1>
    %3 = tosa.reduce_all %2 {axis = 1 : i32} : (tensor<1x42xi1>) -> tensor<1x1xi1>
    %4 = tosa.bitwise_xor %3, %3 : (tensor<1x1xi1>, tensor<1x1xi1>) -> tensor<1x1xi1>
    return %1, %4 : tensor<f32>, tensor<1x1xi1>
  }
}
