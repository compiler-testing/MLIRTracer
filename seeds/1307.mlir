module {
  func.func @main(%arg0: tensor<87xf32>, %arg1: tensor<42x47x55x96x64xi8>, %arg2: tensor<1x1x1x1x1xi8>) -> (tensor<42x47x55x96x64xi8>, tensor<87xf32>) {
    %0 = tosa.reverse %arg0 {axis = 0 : i32} : (tensor<87xf32>) -> tensor<87xf32>
    %1 = tosa.minimum %0, %0 : (tensor<87xf32>, tensor<87xf32>) -> tensor<87xf32>
    %2 = tosa.logical_left_shift %arg1, %arg2 : (tensor<42x47x55x96x64xi8>, tensor<1x1x1x1x1xi8>) -> tensor<42x47x55x96x64xi8>
    %3 = tosa.clamp %1 {min_val = 2.000000e+00 : f32, max_val = 2.100000e+01 : f32} : (tensor<87xf32>) -> tensor<87xf32>
    return %2, %3 : tensor<42x47x55x96x64xi8>, tensor<87xf32>
  }
}
