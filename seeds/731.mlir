module {
  func.func @main(%arg0: tensor<26xf32>, %arg1: tensor<60xi1>) -> (tensor<1xi1>, tensor<1xf32>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<26xf32>) -> tensor<26xf32>
    %1 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<26xf32>) -> tensor<1xf32>
    %2 = tosa.add %1, %1 : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %3 = tosa.clamp %2 {min_val = -3.100000e+01 : f32, max_val = 4.900000e+01 : f32} : (tensor<1xf32>) -> tensor<1xf32>
    %4 = tosa.reduce_any %arg1 {axis = 0 : i32} : (tensor<60xi1>) -> tensor<1xi1>
    %5 = tosa.identity %3 : (tensor<1xf32>) -> tensor<1xf32>
    return %4, %5 : tensor<1xi1>, tensor<1xf32>
  }
}
