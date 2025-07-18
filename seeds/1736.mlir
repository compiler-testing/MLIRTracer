module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<9x69x66x90xf32>, %arg3: tensor<13x98x71x58xf32>, %arg4: tensor<13xf32>) -> (tensor<f32>, tensor<9x236x204x13xf32>, tensor<9x1x204x13xi1>) {
    %0 = tosa.pow %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %input_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %weight_zp_1 = "tosa.const"() <{values = dense<0.0> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = tosa.transpose_conv2d %arg2, %arg3, %arg4, %input_zp_1, %weight_zp_1 {acc_type = f32, out_pad = array<i64: 1, 1, 1, 2>, stride = array<i64: 2, 2>, out_shape = array<i64: 9, 236, 204, 13>} : (tensor<9x69x66x90xf32>, tensor<13x98x71x58xf32>, tensor<13xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<9x236x204x13xf32>
    %2 = tosa.identity %1 : (tensor<9x236x204x13xf32>) -> tensor<9x236x204x13xf32>
    %3 = tosa.reduce_min %2 {axis = 1 : i32} : (tensor<9x236x204x13xf32>) -> tensor<9x1x204x13xf32>
    %4 = tosa.abs %2 : (tensor<9x236x204x13xf32>) -> tensor<9x236x204x13xf32>
    %5 = tosa.greater_equal %3, %3 : (tensor<9x1x204x13xf32>, tensor<9x1x204x13xf32>) -> tensor<9x1x204x13xi1>
    %6 = tosa.reduce_any %5 {axis = 1 : i32} : (tensor<9x1x204x13xi1>) -> tensor<9x1x204x13xi1>
    return %0, %4, %6 : tensor<f32>, tensor<9x236x204x13xf32>, tensor<9x1x204x13xi1>
  }
}
