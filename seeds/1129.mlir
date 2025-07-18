module {
  func.func @main(%arg0: tensor<21xi16>, %arg1: tensor<f32>) -> (tensor<63xi16>, tensor<4xi16>, tensor<i1>, tensor<1xi16>) {
    %0 = tosa.abs %arg0 : (tensor<21xi16>) -> tensor<21xi16>
    %1 = tosa.sigmoid %arg1 : (tensor<f32>) -> tensor<f32>
    %2 = tosa.logical_right_shift %0, %0 : (tensor<21xi16>, tensor<21xi16>) -> tensor<21xi16>
    %t_3 = tosa.const_shape {values = dense<[ 3 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %3 = tosa.tile %2, %t_3 : (tensor<21xi16>, !tosa.shape<1>) -> tensor<63xi16>
    %4 = tosa.equal %1, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = tosa.abs %4 : (tensor<i1>) -> tensor<i1>
    %s_6_start = tosa.const_shape {values = dense<[ 7 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %s_6_size = tosa.const_shape {values = dense<[ 4 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.slice %0, %s_6_start, %s_6_size : (tensor<21xi16>, !tosa.shape<1>, !tosa.shape<1>) -> tensor<4xi16>
    %7 = tosa.log %1 : (tensor<f32>) -> tensor<f32>
    %8 = tosa.greater_equal %7, %7 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %9 = tosa.add %8, %8 : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %10 = tosa.arithmetic_right_shift %9, %5 {round = true} : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %11 = tosa.reduce_min %0 {axis = 0 : i32} : (tensor<21xi16>) -> tensor<1xi16>
    return %3, %6, %10, %11 : tensor<63xi16>, tensor<4xi16>, tensor<i1>, tensor<1xi16>
  }
}
