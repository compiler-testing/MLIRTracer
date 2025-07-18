module {
  func.func @main(%arg0: tensor<f32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<63x18x96xi32>, %arg4: tensor<49x92x86x6xi1>) -> (tensor<i32>, tensor<126x36x288xi32>, tensor<49x1x86x6xi1>, tensor<i1>, tensor<f32>) {
    %0 = tosa.ceil %arg0 : (tensor<f32>) -> tensor<f32>
    %1 = tosa.arithmetic_right_shift %arg1, %arg2 {round = false} : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %t_2 = tosa.const_shape {values = dense<[ 2, 2, 3 ]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %2 = tosa.tile %arg3, %t_2 : (tensor<63x18x96xi32>, !tosa.shape<3>) -> tensor<126x36x288xi32>
    %3 = tosa.reduce_any %arg4 {axis = 1 : i32} : (tensor<49x92x86x6xi1>) -> tensor<49x1x86x6xi1>
    %4 = tosa.log %0 : (tensor<f32>) -> tensor<f32>
    %5 = tosa.greater %4, %0 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = tosa.log %4 : (tensor<f32>) -> tensor<f32>
    return %1, %2, %3, %5, %6 : tensor<i32>, tensor<126x36x288xi32>, tensor<49x1x86x6xi1>, tensor<i1>, tensor<f32>
  }
}
