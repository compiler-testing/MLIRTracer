module {
  func.func @main(%arg0: tensor<90x4x12x52xi1>, %arg1: tensor<1x76x97x47x51x25xf32>) -> (tensor<1x76x97x47x51x25xf32>, tensor<90x1x12x52xi1>, tensor<90x1x1x52xi1>, tensor<270x2x24x1xi1>) {
    %0 = tosa.reduce_all %arg0 {axis = 1 : i32} : (tensor<90x4x12x52xi1>) -> tensor<90x1x12x52xi1>
    %1 = tosa.floor %arg1 : (tensor<1x76x97x47x51x25xf32>) -> tensor<1x76x97x47x51x25xf32>
    %t_2 = tosa.const_shape {values = dense<[ 3, 2, 2, 3 ]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.tile %0, %t_2 : (tensor<90x1x12x52xi1>, !tosa.shape<4>) -> tensor<270x2x24x156xi1>
    %3 = tosa.bitwise_not %2 : (tensor<270x2x24x156xi1>) -> tensor<270x2x24x156xi1>
    %4 = tosa.clamp %1 {min_val = -1.600000e+01 : f32, max_val = 1.250000e+02 : f32} : (tensor<1x76x97x47x51x25xf32>) -> tensor<1x76x97x47x51x25xf32>
    %5 = tosa.tanh %4 : (tensor<1x76x97x47x51x25xf32>) -> tensor<1x76x97x47x51x25xf32>
    %6 = tosa.sub %4, %5 : (tensor<1x76x97x47x51x25xf32>, tensor<1x76x97x47x51x25xf32>) -> tensor<1x76x97x47x51x25xf32>
    %7 = tosa.logical_not %0 : (tensor<90x1x12x52xi1>) -> tensor<90x1x12x52xi1>
    %8 = tosa.reduce_any %3 {axis = 3 : i32} : (tensor<270x2x24x156xi1>) -> tensor<270x2x24x1xi1>
    %9 = tosa.reduce_product %0 {axis = 2 : i32} : (tensor<90x1x12x52xi1>) -> tensor<90x1x1x52xi1>
    %10 = tosa.bitwise_xor %8, %8 : (tensor<270x2x24x1xi1>, tensor<270x2x24x1xi1>) -> tensor<270x2x24x1xi1>
    return %6, %7, %9, %10 : tensor<1x76x97x47x51x25xf32>, tensor<90x1x12x52xi1>, tensor<90x1x1x52xi1>, tensor<270x2x24x1xi1>
  }
}
