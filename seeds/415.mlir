module {
  func.func @main(%arg0: tensor<54x40x51x67x61x82xi32>, %arg1: tensor<54x1x51x67x1x1xi32>, %arg2: tensor<19x17xf32>, %arg3: tensor<15xi1>, %arg4: tensor<15xi1>) -> (tensor<54x40x51x67x61x82xi32>, tensor<1xi1>, tensor<19x1xf32>, tensor<30xi1>, tensor<19x1xf32>) {
    %0 = tosa.add %arg0, %arg1 : (tensor<54x40x51x67x61x82xi32>, tensor<54x1x51x67x1x1xi32>) -> tensor<54x40x51x67x61x82xi32>
    %1 = tosa.tanh %arg2 : (tensor<19x17xf32>) -> tensor<19x17xf32>
    %2 = tosa.logical_xor %arg3, %arg4 : (tensor<15xi1>, tensor<15xi1>) -> tensor<15xi1>
    %t_3 = tosa.const_shape {values = dense<[ 1, 3 ]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.tile %1, %t_3 : (tensor<19x17xf32>, !tosa.shape<2>) -> tensor<19x51xf32>
    %4 = tosa.abs %0 : (tensor<54x40x51x67x61x82xi32>) -> tensor<54x40x51x67x61x82xi32>
    %5 = tosa.reduce_sum %3 {axis = 1 : i32} : (tensor<19x51xf32>) -> tensor<19x1xf32>
    %6 = tosa.reduce_max %2 {axis = 0 : i32} : (tensor<15xi1>) -> tensor<1xi1>
    %7 = tosa.clz %6 : (tensor<1xi1>) -> tensor<1xi1>
    %8 = tosa.clz %2 : (tensor<15xi1>) -> tensor<15xi1>
    %9 = tosa.log %5 : (tensor<19x1xf32>) -> tensor<19x1xf32>
    %10 = tosa.ceil %9 : (tensor<19x1xf32>) -> tensor<19x1xf32>
    %11 = tosa.concat %2, %8 {axis = 0 : i32} : (tensor<15xi1>, tensor<15xi1>) -> tensor<30xi1>
    %12 = tosa.floor %5 : (tensor<19x1xf32>) -> tensor<19x1xf32>
    return %4, %7, %10, %11, %12 : tensor<54x40x51x67x61x82xi32>, tensor<1xi1>, tensor<19x1xf32>, tensor<30xi1>, tensor<19x1xf32>
  }
}
