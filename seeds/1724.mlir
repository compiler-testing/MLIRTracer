module {
  func.func @main(%arg0: tensor<24x8x27x40xf32>, %arg1: tensor<24x1x27x1xf32>, %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<f32>, tensor<24x1x1x2xi1>) {
    %0 = tosa.greater %arg0, %arg1 : (tensor<24x8x27x40xf32>, tensor<24x1x27x1xf32>) -> tensor<24x8x27x40xi1>
    %1 = tosa.reverse %0 {axis = 3 : i32} : (tensor<24x8x27x40xi1>) -> tensor<24x8x27x40xi1>
    %2 = tosa.reduce_all %1 {axis = 3 : i32} : (tensor<24x8x27x40xi1>) -> tensor<24x8x27x1xi1>
    %3 = tosa.reduce_all %2 {axis = 1 : i32} : (tensor<24x8x27x1xi1>) -> tensor<24x1x27x1xi1>
    %4 = tosa.logical_xor %3, %3 : (tensor<24x1x27x1xi1>, tensor<24x1x27x1xi1>) -> tensor<24x1x27x1xi1>
    %5 = tosa.concat %4, %3 {axis = 3 : i32} : (tensor<24x1x27x1xi1>, tensor<24x1x27x1xi1>) -> tensor<24x1x27x2xi1>
    %6 = tosa.reduce_any %5 {axis = 2 : i32} : (tensor<24x1x27x2xi1>) -> tensor<24x1x1x2xi1>
    %7 = tosa.pow %arg2, %arg3 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %8 = tosa.logical_and %6, %6 : (tensor<24x1x1x2xi1>, tensor<24x1x1x2xi1>) -> tensor<24x1x1x2xi1>
    return %7, %8 : tensor<f32>, tensor<24x1x1x2xi1>
  }
}
