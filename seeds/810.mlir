module {
  func.func @main(%arg0: tensor<88x47x38x30x69x27xi16>, %arg1: tensor<88x47x38x30x65x27xi16>, %arg2: tensor<f32>, %arg3: tensor<2x29x75x56xi1>, %arg4: tensor<2x1x75x1xi1>) -> (tensor<88x47x38x30x134x27xi16>, tensor<f32>, tensor<2x29x75x56xi1>, tensor<f32>) {
    %0 = tosa.concat %arg0, %arg1 {axis = 4 : i32} : (tensor<88x47x38x30x69x27xi16>, tensor<88x47x38x30x65x27xi16>) -> tensor<88x47x38x30x134x27xi16>
    %1 = tosa.log %arg2 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.rsqrt %1 : (tensor<f32>) -> tensor<f32>
    %3 = tosa.logical_and %arg3, %arg4 : (tensor<2x29x75x56xi1>, tensor<2x1x75x1xi1>) -> tensor<2x29x75x56xi1>
    %4 = tosa.log %1 : (tensor<f32>) -> tensor<f32>
    return %0, %2, %3, %4 : tensor<88x47x38x30x134x27xi16>, tensor<f32>, tensor<2x29x75x56xi1>, tensor<f32>
  }
}
