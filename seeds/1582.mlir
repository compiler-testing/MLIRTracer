module {
  func.func @main(%arg0: tensor<97xi32>, %arg1: tensor<97xi32>, %arg2: tensor<21x47x14x100x76x94xi1>, %arg3: tensor<81x99xf32>) -> (tensor<21x47x14x100x76x94xi1>, tensor<1xi32>, tensor<81x1xf32>, tensor<81x99xf32>) {
    %0 = tosa.intdiv %arg0, %arg1 : (tensor<97xi32>, tensor<97xi32>) -> tensor<97xi32>
    %1 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<97xi32>) -> tensor<1xi32>
    %2 = tosa.logical_left_shift %1, %1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    %3 = tosa.logical_not %arg2 : (tensor<21x47x14x100x76x94xi1>) -> tensor<21x47x14x100x76x94xi1>
    %4 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %5 = tosa.exp %arg3 : (tensor<81x99xf32>) -> tensor<81x99xf32>
    %6 = tosa.reverse %4 {axis = 0 : i32} : (tensor<1xi32>) -> tensor<1xi32>
    %7 = tosa.reduce_max %5 {axis = 1 : i32} : (tensor<81x99xf32>) -> tensor<81x1xf32>
    %8 = tosa.clamp %5 {min_val = 8.000000e+00 : f32, max_val = 3.400000e+01 : f32} : (tensor<81x99xf32>) -> tensor<81x99xf32>
    return %3, %6, %7, %8 : tensor<21x47x14x100x76x94xi1>, tensor<1xi32>, tensor<81x1xf32>, tensor<81x99xf32>
  }
}
