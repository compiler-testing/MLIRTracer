module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<43x23x27xi16>, %arg3: tensor<13xf32>) -> (tensor<1x23x27xi16>, tensor<i1>, tensor<13xf32>, tensor<26xf32>, tensor<1xf32>, tensor<13xi1>, tensor<13xf32>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %1 = tosa.reduce_sum %arg2 {axis = 0 : i32} : (tensor<43x23x27xi16>) -> tensor<1x23x27xi16>
    %2 = tosa.sigmoid %arg3 : (tensor<13xf32>) -> tensor<13xf32>
    %3 = tosa.logical_not %0 : (tensor<i1>) -> tensor<i1>
    %4 = tosa.reverse %2 {axis = 0 : i32} : (tensor<13xf32>) -> tensor<13xf32>
    %5 = tosa.greater %2, %2 : (tensor<13xf32>, tensor<13xf32>) -> tensor<13xi1>
    %6 = tosa.concat %2, %2 {axis = 0 : i32} : (tensor<13xf32>, tensor<13xf32>) -> tensor<26xf32>
    %7 = tosa.reduce_sum %2 {axis = 0 : i32} : (tensor<13xf32>) -> tensor<1xf32>
    %8 = tosa.logical_right_shift %5, %5 : (tensor<13xi1>, tensor<13xi1>) -> tensor<13xi1>
    %9 = tosa.sigmoid %2 : (tensor<13xf32>) -> tensor<13xf32>
    %10 = tosa.exp %9 : (tensor<13xf32>) -> tensor<13xf32>
    return %1, %3, %4, %6, %7, %8, %10 : tensor<1x23x27xi16>, tensor<i1>, tensor<13xf32>, tensor<26xf32>, tensor<1xf32>, tensor<13xi1>, tensor<13xf32>
  }
}
