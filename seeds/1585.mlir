module {
  func.func @main(%arg0: tensor<21x47x24xi64>, %arg1: tensor<24xi1>, %arg2: tensor<25x94x76x76x8x12xf32>) -> (tensor<1x47x24xi64>, tensor<1xi1>, tensor<25x94x76x76x8x12xf32>) {
    %0 = tosa.reduce_min %arg0 {axis = 0 : i32} : (tensor<21x47x24xi64>) -> tensor<1x47x24xi64>
    %1 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<24xi1>) -> tensor<1xi1>
    %2 = tosa.rsqrt %arg2 : (tensor<25x94x76x76x8x12xf32>) -> tensor<25x94x76x76x8x12xf32>
    return %0, %1, %2 : tensor<1x47x24xi64>, tensor<1xi1>, tensor<25x94x76x76x8x12xf32>
  }
}
