module {
  func.func @main(%arg0: tensor<78x87x99xi8>, %arg1: tensor<78x48x42x35x57xi1>, %arg2: tensor<86x22xf32>) -> (tensor<78x48x42x35x57xi1>, tensor<78x1x1xi1>, tensor<78x1xi32>, tensor<78x1x99xi1>, tensor<78x1xi1>, tensor<313709760xi1>, tensor<86x22xf32>, tensor<1xi1>) {
    %0 = tosa.reduce_sum %arg0 {axis = 1 : i32} : (tensor<78x87x99xi8>) -> tensor<78x1x99xi8>
    %1 = tosa.logical_not %arg1 : (tensor<78x48x42x35x57xi1>) -> tensor<78x48x42x35x57xi1>
    %2 = tosa.logical_not %1 : (tensor<78x48x42x35x57xi1>) -> tensor<78x48x42x35x57xi1>
    %3 = tosa.arithmetic_right_shift %1, %1 {round = false} : (tensor<78x48x42x35x57xi1>, tensor<78x48x42x35x57xi1>) -> tensor<78x48x42x35x57xi1>
    %4 = tosa.argmax %0 {axis = 2 : i32} : (tensor<78x1x99xi8>) -> tensor<78x1xi32>
    %5 = tosa.reduce_sum %0 {axis = 2 : i32} : (tensor<78x1x99xi8>) -> tensor<78x1x1xi8>
    %6 = tosa.clz %2 : (tensor<78x48x42x35x57xi1>) -> tensor<78x48x42x35x57xi1>
    %r_7 = tosa.const_shape {values = dense<[ 313709760 ]> : tensor<1xindex>} : () -> !tosa.shape<1>
    %7 = tosa.reshape %3, %r_7 : (tensor<78x48x42x35x57xi1>, !tosa.shape<1>) -> tensor<313709760xi1>
    %8 = tosa.logical_not %6 : (tensor<78x48x42x35x57xi1>) -> tensor<78x48x42x35x57xi1>
    %9 = tosa.greater %5, %5 : (tensor<78x1x1xi8>, tensor<78x1x1xi8>) -> tensor<78x1x1xi1>
    %10 = tosa.reduce_all %7 {axis = 0 : i32} : (tensor<313709760xi1>) -> tensor<1xi1>
    %11 = tosa.intdiv %4, %4 : (tensor<78x1xi32>, tensor<78x1xi32>) -> tensor<78x1xi32>
    %12 = tosa.reduce_max %10 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %13 = tosa.greater %0, %0 : (tensor<78x1x99xi8>, tensor<78x1x99xi8>) -> tensor<78x1x99xi1>
    %14 = tosa.greater %4, %4 : (tensor<78x1xi32>, tensor<78x1xi32>) -> tensor<78x1xi1>
    %15 = tosa.logical_xor %12, %12 : (tensor<1xi1>, tensor<1xi1>) -> tensor<1xi1>
    %16 = tosa.logical_right_shift %7, %7 : (tensor<313709760xi1>, tensor<313709760xi1>) -> tensor<313709760xi1>
    %17 = tosa.bitwise_not %16 : (tensor<313709760xi1>) -> tensor<313709760xi1>
    %18 = tosa.reduce_min %15 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %19 = tosa.exp %arg2 : (tensor<86x22xf32>) -> tensor<86x22xf32>
    %20 = tosa.reduce_any %18 {axis = 0 : i32} : (tensor<1xi1>) -> tensor<1xi1>
    %21 = tosa.abs %20 : (tensor<1xi1>) -> tensor<1xi1>
    return %8, %9, %11, %13, %14, %17, %19, %21 : tensor<78x48x42x35x57xi1>, tensor<78x1x1xi1>, tensor<78x1xi32>, tensor<78x1x99xi1>, tensor<78x1xi1>, tensor<313709760xi1>, tensor<86x22xf32>, tensor<1xi1>
  }
}
