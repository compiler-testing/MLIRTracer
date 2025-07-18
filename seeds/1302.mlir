module {
  func.func @main(%arg0: tensor<i16>, %arg1: tensor<i16>, %arg2: tensor<10x96x11x13xf32>) -> (tensor<20x288x22x13xf32>, tensor<i16>, tensor<20x96x11x13xf32>) {
    %0 = tosa.arithmetic_right_shift %arg0, %arg1 {round = false} : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %1 = tosa.tanh %arg2 : (tensor<10x96x11x13xf32>) -> tensor<10x96x11x13xf32>
    %2 = tosa.clamp %0 {min_val = 6 : i16, max_val = 46 : i16} : (tensor<i16>) -> tensor<i16>
    %3 = tosa.abs %1 : (tensor<10x96x11x13xf32>) -> tensor<10x96x11x13xf32>
    %t_4 = tosa.const_shape {values = dense<[ 2, 3, 2, 1 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %4 = tosa.tile %3, %t_4 : (tensor<10x96x11x13xf32>, !tosa.shape<4>) -> tensor<20x288x22x13xf32>
    %5 = tosa.bitwise_xor %0, %2 : (tensor<i16>, tensor<i16>) -> tensor<i16>
    %6 = tosa.concat %3, %1 {axis = 0 : i32} : (tensor<10x96x11x13xf32>, tensor<10x96x11x13xf32>) -> tensor<20x96x11x13xf32>
    return %4, %5, %6 : tensor<20x288x22x13xf32>, tensor<i16>, tensor<20x96x11x13xf32>
  }
}
