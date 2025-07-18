module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<81x18x71x60x84x62xi32>, %arg2: tensor<23x62x7xi1>) -> (tensor<81x18x71x60x84x62xi32>, tensor<f32>, tensor<1x62x1xi1>) {
    %0 = tosa.exp %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.bitwise_not %arg1 : (tensor<81x18x71x60x84x62xi32>) -> tensor<81x18x71x60x84x62xi32>
    %2 = tosa.sub %1, %1 : (tensor<81x18x71x60x84x62xi32>, tensor<81x18x71x60x84x62xi32>) -> tensor<81x18x71x60x84x62xi32>
    %3 = tosa.identity %0 : (tensor<f32>) -> tensor<f32>
    %4 = tosa.reduce_all %arg2 {axis = 0 : i32} : (tensor<23x62x7xi1>) -> tensor<1x62x7xi1>
    %5 = tosa.bitwise_not %4 : (tensor<1x62x7xi1>) -> tensor<1x62x7xi1>
    %6 = tosa.logical_and %5, %4 : (tensor<1x62x7xi1>, tensor<1x62x7xi1>) -> tensor<1x62x7xi1>
    %7 = tosa.bitwise_and %2, %2 : (tensor<81x18x71x60x84x62xi32>, tensor<81x18x71x60x84x62xi32>) -> tensor<81x18x71x60x84x62xi32>
    %8 = tosa.exp %3 : (tensor<f32>) -> tensor<f32>
    %9 = tosa.reduce_sum %6 {axis = 2 : i32} : (tensor<1x62x7xi1>) -> tensor<1x62x1xi1>
    return %7, %8, %9 : tensor<81x18x71x60x84x62xi32>, tensor<f32>, tensor<1x62x1xi1>
  }
}
