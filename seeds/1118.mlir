module {
  func.func @main(%arg0: tensor<14x64x81x66xi8>, %arg1: tensor<14x64x1x66xi8>, %arg2: tensor<f32>) -> (tensor<i1>, tensor<14x64x1x66xi1>, tensor<64x1x66xi32>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<14x64x81x66xi8>, tensor<14x64x1x66xi8>) -> tensor<14x64x81x66xi1>
    %1 = tosa.add %0, %0 : (tensor<14x64x81x66xi1>, tensor<14x64x81x66xi1>) -> tensor<14x64x81x66xi1>
    %2 = tosa.bitwise_and %1, %0 : (tensor<14x64x81x66xi1>, tensor<14x64x81x66xi1>) -> tensor<14x64x81x66xi1>
    %3 = tosa.reverse %2 {axis = 0 : i32} : (tensor<14x64x81x66xi1>) -> tensor<14x64x81x66xi1>
    %4 = tosa.log %arg2 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.logical_or %3, %3 : (tensor<14x64x81x66xi1>, tensor<14x64x81x66xi1>) -> tensor<14x64x81x66xi1>
    %6 = tosa.reduce_sum %5 {axis = 2 : i32} : (tensor<14x64x81x66xi1>) -> tensor<14x64x1x66xi1>
    %7 = tosa.bitwise_xor %6, %6 : (tensor<14x64x1x66xi1>, tensor<14x64x1x66xi1>) -> tensor<14x64x1x66xi1>
    %8 = tosa.greater %4, %4 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = tosa.reduce_any %2 {axis = 2 : i32} : (tensor<14x64x81x66xi1>) -> tensor<14x64x1x66xi1>
    %10 = tosa.argmax %7 {axis = 0 : i32} : (tensor<14x64x1x66xi1>) -> tensor<64x1x66xi32>
    %11 = tosa.reverse %10 {axis = 1 : i32} : (tensor<64x1x66xi32>) -> tensor<64x1x66xi32>
    return %8, %9, %11 : tensor<i1>, tensor<14x64x1x66xi1>, tensor<64x1x66xi32>
  }
}
