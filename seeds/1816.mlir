module {
  func.func @main(%arg0: tensor<6x40xf32>, %arg1: tensor<21x47x64x35xi1>, %arg2: tensor<21x47x1x1xi1>) -> (tensor<6x40xf32>, tensor<21x47x1x35xi1>, tensor<34545xi1>) {
    %0 = tosa.ceil %arg0 : (tensor<6x40xf32>) -> tensor<6x40xf32>
    %1 = tosa.bitwise_or %arg1, %arg2 : (tensor<21x47x64x35xi1>, tensor<21x47x1x1xi1>) -> tensor<21x47x64x35xi1>
    %2 = tosa.add %1, %1 : (tensor<21x47x64x35xi1>, tensor<21x47x64x35xi1>) -> tensor<21x47x64x35xi1>
    %3 = tosa.reduce_sum %2 {axis = 2 : i32} : (tensor<21x47x64x35xi1>) -> tensor<21x47x1x35xi1>
    %4 = tosa.clz %3 : (tensor<21x47x1x35xi1>) -> tensor<21x47x1x35xi1>
    %5 = tosa.logical_not %3 : (tensor<21x47x1x35xi1>) -> tensor<21x47x1x35xi1>
    %r_6 = tosa.const_shape {values = dense<[ 34545 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %6 = tosa.reshape %4, %r_6 : (tensor<21x47x1x35xi1>, !tosa.shape<1>) -> tensor<34545xi1>
    return %0, %5, %6 : tensor<6x40xf32>, tensor<21x47x1x35xi1>, tensor<34545xi1>
  }
}
