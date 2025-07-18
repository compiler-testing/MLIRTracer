module {
  func.func @main(%arg0: tensor<32x69x6xi8>, %arg1: tensor<f32>) -> (tensor<32x69x6xi8>, tensor<f32>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<32x69x6xi8>) -> tensor<32x69x6xi8>
    %1 = tosa.reverse %0 {axis = 1 : i32} : (tensor<32x69x6xi8>) -> tensor<32x69x6xi8>
    %2 = tosa.sigmoid %arg1 : (tensor<f32>) -> tensor<f32>
    return %1, %2 : tensor<32x69x6xi8>, tensor<f32>
  }
}
