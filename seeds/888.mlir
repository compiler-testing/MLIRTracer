module {
  func.func @main(%arg0: tensor<31x54x68xf32>, %arg1: tensor<31x100xi8>, %arg2: tensor<1x100xi8>) -> (tensor<31x54x136xf32>, tensor<200xi32>, tensor<200xi32>) {
    %0 = tosa.clamp %arg0 {min_val = 1.700000e+01 : f32, max_val = 5.000000e+01 : f32} : (tensor<31x54x68xf32>) -> tensor<31x54x68xf32>
    %1 = tosa.add %0, %0 : (tensor<31x54x68xf32>, tensor<31x54x68xf32>) -> tensor<31x54x68xf32>
    %2 = tosa.logical_right_shift %arg1, %arg2 : (tensor<31x100xi8>, tensor<1x100xi8>) -> tensor<31x100xi8>
    %3 = tosa.clamp %2 {min_val = -28 : i8, max_val = 17 : i8} : (tensor<31x100xi8>) -> tensor<31x100xi8>
    %4 = tosa.concat %1, %0 {axis = 2 : i32} : (tensor<31x54x68xf32>, tensor<31x54x68xf32>) -> tensor<31x54x136xf32>
    %5 = tosa.argmax %3 {axis = 0 : i32} : (tensor<31x100xi8>) -> tensor<100xi32>
    %6 = tosa.sub %5, %5 : (tensor<100xi32>, tensor<100xi32>) -> tensor<100xi32>
    %7 = tosa.concat %6, %5 {axis = 0 : i32} : (tensor<100xi32>, tensor<100xi32>) -> tensor<200xi32>
    %8 = tosa.bitwise_or %7, %7 : (tensor<200xi32>, tensor<200xi32>) -> tensor<200xi32>
    %9 = tosa.logical_right_shift %8, %7 : (tensor<200xi32>, tensor<200xi32>) -> tensor<200xi32>
    %10 = tosa.clz %7 : (tensor<200xi32>) -> tensor<200xi32>
    return %4, %9, %10 : tensor<31x54x136xf32>, tensor<200xi32>, tensor<200xi32>
  }
}
