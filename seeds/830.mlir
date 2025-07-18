module {
  func.func @main(%arg0: tensor<8x87xf32>, %arg1: tensor<1x87xf32>, %arg2: tensor<26x56x61xi64>, %arg3: tensor<1x1x61xi64>) -> (tensor<174xi32>, tensor<56x61xi32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<8x87xf32>, tensor<1x87xf32>) -> tensor<8x87xf32>
    %1 = tosa.sigmoid %0 : (tensor<8x87xf32>) -> tensor<8x87xf32>
    %2 = tosa.log %1 : (tensor<8x87xf32>) -> tensor<8x87xf32>
    %3 = tosa.clamp %2 {min_val = 4.700000e+01 : f32, max_val = 1.310000e+02 : f32} : (tensor<8x87xf32>) -> tensor<8x87xf32>
    %4 = tosa.logical_left_shift %arg2, %arg3 : (tensor<26x56x61xi64>, tensor<1x1x61xi64>) -> tensor<26x56x61xi64>
    %5 = tosa.concat %3, %0 {axis = 1 : i32} : (tensor<8x87xf32>, tensor<8x87xf32>) -> tensor<8x174xf32>
    %6 = tosa.argmax %5 {axis = 0 : i32} : (tensor<8x174xf32>) -> tensor<174xi32>
    %7 = tosa.argmax %4 {axis = 0 : i32} : (tensor<26x56x61xi64>) -> tensor<56x61xi32>
    return %6, %7 : tensor<174xi32>, tensor<56x61xi32>
  }
}
